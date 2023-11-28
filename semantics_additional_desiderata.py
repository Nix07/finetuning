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
path = "/media/local_nikhil/disk/weights_naive/possessed-candle-14"
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path).to(device)

tokenizer.pad_token_id = tokenizer.eos_token_id
print("Model loaded.")

num_boxes = 7
batch_size = 16
data_file_path = f"/data/nikhil_prakash/anima-2.0/box_datasets/no_instructions/alternative/Random/{num_boxes}/train.jsonl"
object_file_path = (
    "/data/nikhil_prakash/anima-2.0/box_datasets/filtered_objects_with_bnc_frequency.csv"
)

desiderata_methods = {
    "raw_text_start": add_raw_text_at_start,
    "raw_text_end": add_raw_text_at_end,
    "additional_tokens_btw_obj_and_box": additional_token_btw_box_and_object,
    "add_segment_start": add_segment_at_start,
    "add_segment_end": add_segment_at_end,
    "add_boxes_before_correct_segment": add_box_before_correct_segment,
    "incorrect_box_segment_index": diff_index_query_box,
    "Box_object_altered_order": box_object_altered_order,
    "object_not_in_box": alter_box_object_association,
    "no_comma": remove_comma_desiderata,
    "comma_after_object": add_comma_after_object,
}

with open(
    f"/data/nikhil_prakash/anima-2.0/new_masks/llama-7b/heads_affect_direct_logit/positional/0.01.txt",
    "r",
) as f:
    data = f.readlines()
    heads = json.loads(data[0].split(": ")[1])
patching_heads = {0: heads}


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
            # pos = inputs.size(1) - rel_pos - 1
            for head in curr_layer_heads:
                inputs[:, -1, head] = cache[:, -1, head]

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


results = defaultdict(list)
for desiderata in desiderata_methods.keys():
    for _ in tqdm(range(10)):
        raw_data = desiderata_methods[desiderata](
            tokenizer=tokenizer,
            num_samples=500,
            data_file=data_file_path,
        )
        print(f"Desiderata: {desiderata}")
        print(f"Original: {tokenizer.decode(raw_data[0][0])}")
        print(f"Alternate: {tokenizer.decode(raw_data[2][0])}")
        print(f"Label: {tokenizer.decode(raw_data[4][0])}")

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
            ) as _:
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
        print(f"Desiderata: {desiderata}, Accuracy: {acc}")
        results[desiderata].append(acc)

        # Store results in json file
        with open("additional_desiderata_results.json", "w") as f:
            json.dump(results, f, indent=4)
