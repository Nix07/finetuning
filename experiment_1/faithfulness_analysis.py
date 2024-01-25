import json
import random
import fire
import numpy as np
from functools import partial
from tqdm import tqdm
from collections import defaultdict

import torch
import transformers
from torch.utils.data import DataLoader

from pp_utils import (
    get_model_and_tokenizer,
    load_eval_data,
    get_mean_activations,
    compute_topk_components,
    eval_circuit_performance,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_circuit(model, circuit_heads):
    circuit_components = {}
    circuit_components[0] = defaultdict(list)
    circuit_components[2] = defaultdict(list)
    circuit_components[-1] = defaultdict(list)

    for layer_idx, head in circuit_heads["value_fetcher"]:
        if model.config.architectures[0] == "LlamaForCausalLM":
            layer = f"model.layers.{layer_idx}.self_attn.o_proj"
        else:
            layer = f"base_model.model.model.layers.{layer_idx}.self_attn.o_proj"
        circuit_components[0][layer].append(head)

    for layer_idx, head in circuit_heads["pos_transmitter"]:
        if model.config.architectures[0] == "LlamaForCausalLM":
            layer = f"model.layers.{layer_idx}.self_attn.o_proj"
        else:
            layer = f"base_model.model.model.layers.{layer_idx}.self_attn.o_proj"
        circuit_components[0][layer].append(head)

    for layer_idx, head in circuit_heads["pos_detector"]:
        if model.config.architectures[0] == "LlamaForCausalLM":
            layer = f"model.layers.{layer_idx}.self_attn.o_proj"
        else:
            layer = f"base_model.model.model.layers.{layer_idx}.self_attn.o_proj"
        circuit_components[2][layer].append(head)

    for layer_idx, head in circuit_heads["struct_reader"]:
        if model.config.architectures[0] == "LlamaForCausalLM":
            layer = f"model.layers.{layer_idx}.self_attn.o_proj"
        else:
            layer = f"base_model.model.model.layers.{layer_idx}.self_attn.o_proj"
        circuit_components[-1][layer].append(head)

    return circuit_components


def negative_heads_impact(
    root_path="/data/nikhil_prakash/anima-2.0",
    model_name="llama",
    datafile="data/dataset.jsonl",
    num_samples=500,
    batch_size=50,
    circuit_name="llama",
    head_group="value_fetcher",
    interval=1,
):
    datafile = f"{root_path}/{datafile}"
    model, tokenizer = get_model_and_tokenizer(model_name)

    dataloader = load_eval_data(
        tokenizer=tokenizer,
        datafile=datafile,
        num_samples=num_samples,
        batch_size=batch_size,
    )

    with open(
        f"{root_path}/experiment_1/results/circuits/{circuit_name}_circuit.json",
        "r",
        encoding="utf-8",
    ) as f:
        circuit = json.load(f)

    mean_activations, modules = get_mean_activations(
        model, tokenizer, datafile, num_samples=500, batch_size=50
    )

    patching_score = torch.load(
        f"{root_path}/experiment_1/results/path_patching/{circuit_name}_circuit/{head_group}.pt"
    )
    ordered_heads = compute_topk_components(
        patching_score,
        k=model.config.num_hidden_layers * model.config.num_attention_heads,
        largest=False,
    )

    results = defaultdict(list)
    altered_circuit = circuit.copy()
    for idx in tqdm(
        range(
            0,
            model.config.num_hidden_layers * model.config.num_attention_heads,
            interval,
        ),
        desc="#Heads",
    ):
        for group in circuit.keys():
            if group == head_group:
                altered_circuit[group] = ordered_heads[: idx + 1]

        circuit_components = get_circuit(model, altered_circuit)

        circuit_acc = eval_circuit_performance(
            model, dataloader, modules, circuit_components, mean_activations
        )
        print(f"{head_group} | {idx+1} | {circuit_acc}\n")

        results[head_group].append(circuit_acc)

        # Save results in a json file
        with open(
            f"{root_path}/experiment_1/results/{model_name}_{head_group}_heads.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(results, f, indent=4)


if __name__ == "__main__":
    fire.Fire(negative_heads_impact)
