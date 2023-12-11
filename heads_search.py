import torch
import json
import transformers
from transformers import (
    AutoModelForCausalLM,
    LlamaTokenizer,
)
from functools import partial
from baukit import TraceDict
from einops import rearrange, einsum
from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm
from peft import PeftModel
import pickle
import fire

from analysis_utils import *
from counterfactual_datasets.entity_tracking import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed = 20
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
transformers.set_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def head_search(
    model_name: str,
    dataset_path: str,
    root_path: str,
    batch_size: int = 50,
):
    print(f"Loading model {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = LlamaTokenizer.from_pretrained(
        "hf-internal-testing/llama-tokenizer", padding_side="right"
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    print(f"Loading dataset {dataset_path}")
    raw_data = entity_tracking_example_sampler(
        tokenizer=tokenizer,
        num_samples=500,
        data_file=dataset_path,
        architecture="LLaMAForCausalLM",
    )
    dataset = Dataset.from_dict(
        {
            "input_ids": raw_data[0],
            "last_token_indices": raw_data[1],
            "labels": raw_data[2],
        }
    ).with_format("torch")

    print(f"Length of dataset: {len(dataset)}")

    dataloader = DataLoader(dataset, batch_size=batch_size)

    if model.config.architectures[0] == "LlamaForCausalLM":
        modules = [f"model.layers.{layer}.self_attn.o_proj" for layer in range(model.config.num_attention_heads)]
    else:
        modules = [
            f"base_model.model.model.layers.{layer}.self_attn.o_proj"
            for layer in range(model.config.num_attention_heads)
        ]
    mean_activations = get_mean_activations(
        model, tokenizer, modules, dataset_path, batch_size
    )

    results = defaultdict(dict)
    for n_value_fetcher in tqdm(range(20, 55, 5), desc="value_fetcher"):
        for n_pos_trans in range(5, 25, 5):
            for n_pos_detect in range(5, 35, 5):
                for n_struct_read in range(0, 5, 5):
                    circuit_components = {}
                    circuit_components[0] = defaultdict(list)
                    circuit_components[2] = defaultdict(list)
                    circuit_components[-1] = defaultdict(list)
                    circuit_components[-2] = defaultdict(list)

                    path = root_path + "/direct_logit_heads.pt"

                    direct_logit_heads, _ = compute_topk_components(
                        torch.load(path), k=n_value_fetcher, largest=False
                    )

                    path = root_path + "/heads_affect_direct_logit.pt"
                    heads_affecting_direct_logit_heads, _ = compute_topk_components(
                        torch.load(path), k=n_pos_trans, largest=False
                    )

                    path = root_path + "/heads_at_query_box_pos.pt"
                    head_at_query_box_token, _ = compute_topk_components(
                        torch.load(path), k=n_pos_detect, largest=False
                    )

                    path = root_path + "/heads_at_prev_query_box_pos.pt"
                    heads_at_prev_box_pos, _ = compute_topk_components(
                        torch.load(path), k=n_struct_read, largest=False
                    )

                    intersection = []
                    for head in direct_logit_heads:
                        if head in heads_affecting_direct_logit_heads:
                            intersection.append(head)

                    for head in intersection:
                        direct_logit_heads.remove(head)

                    print(
                        len(direct_logit_heads),
                        len(heads_affecting_direct_logit_heads),
                        len(head_at_query_box_token),
                        len(heads_at_prev_box_pos),
                    )

                    for layer_idx, head in direct_logit_heads:
                        if model.config.architectures[0] == "LlamaForCausalLM":
                            layer = f"model.layers.{layer_idx}.self_attn.o_proj"
                        else:
                            layer = f"base_model.model.model.layers.{layer_idx}.self_attn.o_proj"
                        circuit_components[0][layer].append(head)

                    for layer_idx, head in heads_affecting_direct_logit_heads:
                        if model.config.architectures[0] == "LlamaForCausalLM":
                            layer = f"model.layers.{layer_idx}.self_attn.o_proj"
                        else:
                            layer = f"base_model.model.model.layers.{layer_idx}.self_attn.o_proj"
                        circuit_components[0][layer].append(head)

                    for layer_idx, head in head_at_query_box_token:
                        if model.config.architectures[0] == "LlamaForCausalLM":
                            layer = f"model.layers.{layer_idx}.self_attn.o_proj"
                        else:
                            layer = f"base_model.model.model.layers.{layer_idx}.self_attn.o_proj"
                        circuit_components[2][layer].append(head)

                    for layer_idx, head in heads_at_prev_box_pos:
                        if model.config.architectures[0] == "LlamaForCausalLM":
                            layer = f"model.layers.{layer_idx}.self_attn.o_proj"
                        else:
                            layer = f"base_model.model.model.layers.{layer_idx}.self_attn.o_proj"
                        circuit_components[-1][layer].append(head)

                    for pos in circuit_components.keys():
                        for layer_idx in circuit_components[pos].keys():
                            circuit_components[pos][layer_idx] = list(
                                set(circuit_components[pos][layer_idx])
                            )

                    results[
                        str(
                            (
                                len(direct_logit_heads),
                                len(heads_affecting_direct_logit_heads),
                                len(head_at_query_box_token),
                                len(heads_at_prev_box_pos),
                            )
                        )
                    ] = eval(
                        model, dataloader, modules, circuit_components, mean_activations
                    )

                    with open("num_heads_results.json", "w") as file:
                        json.dump(results, file, indent=4)


if __name__ == "__main__":
    fire.Fire(head_search)
