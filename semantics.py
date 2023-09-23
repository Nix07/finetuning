import torch
import os
import json
from torch.nn import CosineSimilarity
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, LlamaTokenizer
from functools import partial
from baukit import TraceDict
from einops import rearrange, einsum
from collections import defaultdict
import matplotlib.pyplot as plt
from plotly_utils import imshow, scatter
from tqdm import tqdm
from datasets import Dataset
from torch.utils.data import DataLoader
import math
import seaborn as sns
from peft import PeftModel
import pickle

import pysvelte
import analysis_utils
from counterfactual_datasets.entity_tracking import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

print("Loading model...")
path = "./llama_7b/"
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path).to(device)

# base_model = "decapoda-research/llama-7b-hf"
# lora_weights = "tiedong/goat-lora-7b"

# tokenizer = LlamaTokenizer.from_pretrained(
#     "hf-internal-testing/llama-tokenizer", padding_side="right"
# )
# model = LlamaForCausalLM.from_pretrained(
#     base_model,
#     load_in_8bit=False,
#     torch_dtype=torch.float32,
#     device_map="auto",
# )
# model = PeftModel.from_pretrained(
#     model,
#     lora_weights,
#     torch_dtype=torch.float32,
#     device_map={"": 0},
# )


tokenizer.pad_token_id = tokenizer.eos_token_id
print("Model loaded.")

relative_pos = {
    "heads_affect_direct_logit": 0,
    "heads_at_query_box_pos": 2,
    "heads_at_prev_query_box_pos": -1,
}

num_boxes = 7
batch_size = 8
data_file_path = f"./box_datasets/no_instructions/alternative/Random/{num_boxes}/train.jsonl"
object_file_path = "./box_datasets/filtered_objects_with_bnc_frequency.csv"

desiderata_methods = {
    "positional": positional_desiderata,
    "object_value": object_value_desiderata,
    "box_label_value": box_label_value_desiderata,
}


def load_data(raw_data, batch_size):
    dataset = Dataset.from_dict(
        {
            "base_input_ids": raw_data[0],
            "base_input_last_pos": raw_data[1],
            "source_input_ids": raw_data[2],
            "source_input_last_pos": raw_data[3],
            "labels": raw_data[4],
        }
    ).with_format("torch")
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return dataloader


def patching(inputs, output, layer, patching_heads, bi, input_tokens):
    if isinstance(inputs, tuple):
        inputs = inputs[0]

    inputs = rearrange(
        inputs,
        "batch seq_len (n_heads d_head) -> batch seq_len n_heads d_head",
        n_heads=model.config.num_attention_heads,
    )

    cache = rearrange(
        source_cache[bi][layer],
        "batch seq_len (n_heads d_head) -> batch seq_len n_heads d_head",
        n_heads=model.config.num_attention_heads,
    )

    for rel_pos in patching_heads.keys():
        if model.config.architectures[0] == "LlamaForCausalLM":
            layer_index = int(layer.split(".")[2])
        else:
            layer_index = int(layer.split(".")[4])
        curr_layer_heads = [h for l, h in patching_heads[rel_pos] if l == layer_index]

        if rel_pos == -1:
            for batch in range(inputs.size(0)):
                prev_query_box_pos = analysis_utils.compute_prev_query_box_pos(
                    input_tokens["base_input_ids"][batch],
                    input_tokens["base_input_last_pos"][batch],
                )
                for head in curr_layer_heads:
                    inputs[batch, prev_query_box_pos, head] = cache[batch, prev_query_box_pos, head]
        else:
            pos = inputs.size(1) - rel_pos - 1
            for head in curr_layer_heads:
                inputs[:, pos, head] = cache[:, pos, head]

    inputs = rearrange(
        inputs,
        "batch seq_len n_heads d_head -> batch seq_len (n_heads d_head)",
        n_heads=model.config.num_attention_heads,
    )

    w_o = model.state_dict()[f"{layer}.weight"]
    output = einsum(
        inputs, w_o, "batch seq_len hidden_size, d_model hidden_size -> batch seq_len d_model"
    )

    del w_o
    torch.cuda.empty_cache()
    return output


results = {}
results["llama"] = {}

for head_group in [
    "heads_affect_direct_logit",
    "heads_at_query_box_pos",
    "heads_at_prev_query_box_pos",
]:
    if head_group not in results["llama"]:
        results["llama"][head_group] = defaultdict(list)

    for desiderata in ["positional", "object_value", "box_label_value"]:
        with open(f"./new_masks/llama-7b/{head_group}/{desiderata}/0.01.txt", "r") as f:
            data = f.readlines()
            heads = json.loads(data[0].split(": ")[1])

        patching_heads = {relative_pos[head_group]: heads}

        for _ in tqdm(range(10)):
            raw_data = desiderata_methods[desiderata](
                tokenizer=tokenizer,
                num_samples=1000,
                data_file=data_file_path,
                object_file=object_file_path,
                num_boxes=7,
                alt_format=True,
                correct_pred_indices=[],
            )

            dataloader = load_data(raw_data=raw_data, batch_size=batch_size)

            if model.config.architectures[0] == "LlamaForCausalLM":
                modules = [f"model.layers.{i}.self_attn.o_proj" for i in range(32)]
            else:
                modules = [f"base_model.model.model.layers.{i}.self_attn.o_proj" for i in range(32)]

            source_cache = {}
            for bi, inputs in enumerate(dataloader):
                for k, v in inputs.items():
                    if v is not None and isinstance(v, torch.Tensor):
                        inputs[k] = v.to(model.device)

                with TraceDict(model, modules, retain_input=True) as cache:
                    _ = model(inputs["source_input_ids"])

                for module in modules:
                    if bi in source_cache:
                        source_cache[bi][module] = cache[module].input.detach().cpu()
                    else:
                        source_cache[bi] = {module: cache[module].input.detach().cpu()}

            correct_count, total_count = 0, 0
            for bi, inputs in tqdm(enumerate(dataloader)):
                for k, v in inputs.items():
                    if v is not None and isinstance(v, torch.Tensor):
                        inputs[k] = v.to(model.device)

                with TraceDict(
                    model,
                    modules,
                    retain_input=True,
                    edit_output=partial(
                        patching, patching_heads=patching_heads, bi=bi, input_tokens=inputs
                    ),
                ) as cache:
                    outputs = model(inputs["base_input_ids"])

                for idx in range(inputs["base_input_ids"].size(0)):
                    label = inputs["labels"][idx].item()
                    pred = torch.argmax(outputs.logits[idx, -1], dim=-1).item()

                    if label == pred:
                        correct_count += 1
                    total_count += 1

                del outputs
                torch.cuda.empty_cache()

            acc = round(correct_count / total_count * 100, 2)
            print(f"Head group: {head_group}, Desiderata: {desiderata}, Accuracy: {acc}")
            results["llama"][head_group][desiderata].append(acc)

            # Store results in json file
            with open("llama_semantic_results.json", "w") as f:
                json.dump(results, f, indent=4)
