import os
import random
import math
import sys
import numpy as np
from functools import partial
from collections import defaultdict

import torch
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
)
from baukit import TraceDict, nethook
from einops import rearrange, einsum
from peft import PeftModel
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

curr_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.abspath(os.path.join(curr_dir, os.pardir))
sys.path.append(parent_dir)
from data.data_utils import (
    load_pp_data,
    sample_box_data,
    get_data_for_mean_ablation,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_topk_components(
    patching_scores: torch.Tensor, k: int, largest=True, return_values=False
):
    """
    Computes the topk most influential components (i.e. heads) for patching.

    Args:
        patching_scores: patching scores for the components.
        k: number of components to return.
        largest: whether to return the largest or smallest components.
        return_values: whether to return the values of the components or not.
    """

    top_indices = torch.topk(patching_scores.flatten(), k, largest=largest).indices
    top_values = torch.topk(patching_scores.flatten(), k, largest=largest).values

    # Convert the top_indices to 2D indices
    row_indices = top_indices // patching_scores.shape[1]
    col_indices = top_indices % patching_scores.shape[1]
    top_components = torch.stack((row_indices, col_indices), dim=1)
    # Get the top indices as a list of 2D indices (row, column)
    top_components = top_components.tolist()

    if return_values:
        return top_components, top_values.tolist()
    else:
        return top_components


def compute_prev_query_box_pos(input_ids, last_token_index):
    """
    Computes the position of the previous query box label token.

    Args:
        input_ids: input ids of the example.
        last_token_index: last token index of the example.
    """

    query_box_token = input_ids[last_token_index - 2]
    prev_query_box_token_pos = (
        (input_ids[: last_token_index - 2] == query_box_token).nonzero().item()
    )
    return prev_query_box_token_pos


def get_model_and_tokenizer(model_name: str):
    """
    Loads the model and tokenizer.

    Args:
        model_name (str): Name of the model to load.
    """

    tokenizer = LlamaTokenizer.from_pretrained(
        "hf-internal-testing/llama-tokenizer", padding_side="right"
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    if model_name == "llama":
        # path = "/data/nikhil_prakash/llama_weights/7B/"
        path = "/home/local_nikhil/Projects/llama_weights/7B/"
        model = LlamaForCausalLM.from_pretrained(path).to(device)

    elif model_name == "goat":
        base_model = "decapoda-research/llama-7b-hf"
        lora_weights = "tiedong/goat-lora-7b"
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=False,
            torch_dtype=torch.float32,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float32,
            device_map={"": 0},
        )

    elif model_name == "vicuna":
        path = "AlekseyKorshuk/vicuna-7b"
        model = LlamaForCausalLM.from_pretrained(path).to(device)

    elif model_name == "float":
        path = "nikhil07prakash/float-7b"
        model = LlamaForCausalLM.from_pretrained(path).to(device)

    return model, tokenizer


def load_dataloader(
    model: LlamaForCausalLM,
    tokenizer: LlamaTokenizer,
    datafile: str,
    num_samples: int,
    num_boxes: int,
    batch_size: int,
):
    """
    Loads the data (original and counterfactual) from the datafile and creates a dataloader.

    Args:
        tokenizer: tokenizer to use.
        datafile: path to the datafile.
        num_samples: number of samples to use from the datafile.
        num_boxes: number of boxes in the datafile.
        batch_size: batch size to use for the dataloader.
    """
    raw_data = load_pp_data(
        model=model,
        tokenizer=tokenizer,
        num_samples=num_samples,
        data_file=datafile,
        num_boxes=num_boxes,
    )
    base_tokens = raw_data[0]  # Clean inputs
    base_last_token_indices = raw_data[1]  # Clean last token indices
    source_tokens = raw_data[2]  # Corrupt inputs
    source_last_token_indices = raw_data[3]  # Corrupt last token indices
    correct_answer_token = raw_data[4]  # Correct answer token

    dataset = Dataset.from_dict(
        {
            "base_tokens": base_tokens,
            "base_last_token_indices": base_last_token_indices,
            "source_tokens": source_tokens,
            "source_last_token_indices": source_last_token_indices,
            "labels": correct_answer_token,
        }
    ).with_format("numpy")
    dataloader = DataLoader(dataset, batch_size=batch_size)

    return dataloader


def get_caches(
    model: LlamaForCausalLM,
    dataloader: torch.utils.data.DataLoader,
):
    """
    Computes the clean and corrupt caches for the model.

    Args:
        model: model to compute the caches for.
        dataloader: dataloader containing clean and corrupt inputs.
    """

    if model.config.architectures[0] == "LlamaForCausalLM":
        hook_points = [
            f"model.layers.{layer}.self_attn.o_proj"
            for layer in range(model.config.num_hidden_layers)
        ]
    else:
        hook_points = [
            f"base_model.model.model.layers.{layer}.self_attn.o_proj"
            for layer in range(model.config.num_hidden_layers)
        ]

    apply_softmax = torch.nn.Softmax(dim=-1)
    clean_cache, corrupt_cache = {}, {}
    clean_logit_outputs, corrupt_logit_outputs = defaultdict(dict), defaultdict(dict)

    with torch.no_grad():
        for bi, inp in tqdm(enumerate(dataloader), desc="Clean cache"):
            batch_size = inp["base_tokens"].size(0)

            for k, v in inp.items():
                if v is not None and isinstance(v, torch.Tensor):
                    inp[k] = v.to(model.device)

            with TraceDict(
                model,
                hook_points,
                retain_input=True,
            ) as cache:
                output = model(inp["base_tokens"])

            for k, v in cache.items():
                if v is not None and isinstance(v, nethook.Trace):
                    cache[k].input = v.input.to("cpu")
                    cache[k].output = v.output.to("cpu")

            clean_cache[bi] = cache
            for i in range(batch_size):
                logits = apply_softmax(
                    output.logits[i, inp["base_last_token_indices"][i]]
                )
                clean_logit_outputs[bi][i] = (logits[inp["labels"][i]]).item()

            del output, cache, logits, inp
            torch.cuda.empty_cache()
        print("CLEAN CACHE COMPUTED")

        for bi, inp in tqdm(enumerate(dataloader), desc="Corrupt cache"):
            for k, v in inp.items():
                if v is not None and isinstance(v, torch.Tensor):
                    inp[k] = v.to(model.device)

            with TraceDict(
                model,
                hook_points,
                retain_input=True,
            ) as cache:
                output = model(inp["source_tokens"])

            for k, v in cache.items():
                if v is not None and isinstance(v, nethook.Trace):
                    cache[k].input = v.input.to("cpu")
                    cache[k].output = v.output.to("cpu")

            corrupt_cache[bi] = cache
            for i in range(batch_size):
                logits = apply_softmax(
                    output.logits[i, inp["source_last_token_indices"][i]]
                )
                corrupt_logit_outputs[bi][i] = (logits[inp["labels"][i]]).item()

            del output, cache, logits, inp
            torch.cuda.empty_cache()
        print("CORRUPT CACHE COMPUTED")

    return (
        clean_cache,
        corrupt_cache,
        clean_logit_outputs,
        corrupt_logit_outputs,
        hook_points,
    )


def patching_sender_heads(
    inputs=None,
    output=None,
    layer: str = None,
    model: LlamaForCausalLM = None,
    clean_cache: dict = None,
    corrupt_cache: dict = None,
    base_tokens: list = None,
    sender_layer: str = None,
    sender_head: str = None,
    clean_last_token_indices: list = None,
    corrupt_last_token_indices: list = None,
    rel_pos: int = None,
    batch_size: int = None,
):
    """
    Patches the output of the sender head and stores the input of the receiver head.

    Args:
        inputs: inputs to the layer.
        output: output of the layer.
        layer: layer to patch.
        model: model to patch.
        clean_cache: clean cache of the model.
        corrupt_cache: corrupt cache of the model.
        base_tokens: clean inputs.
        sender_layer: layer of the sender head.
        sender_head: sender head.
        clean_last_token_indices: clean last token indices.
        corrupt_last_token_indices: corrupt last token indices.
        rel_pos: relative position of the query box label token.
        batch_size: batch size of the dataloader.
    """

    input = inputs[0]

    if "o_proj" in layer:
        input = rearrange(
            input,
            "batch seq_len (n_heads d_head) -> batch seq_len n_heads d_head",
            n_heads=model.config.num_attention_heads,
        )
        clean_head_outputs = rearrange(
            clean_cache[layer].input,
            "batch seq_len (n_heads d_head) -> batch seq_len n_heads d_head",
            n_heads=model.config.num_attention_heads,
        )
        corrupt_head_outputs = rearrange(
            corrupt_cache[layer].input,
            "batch seq_len (n_heads d_head) -> batch seq_len n_heads d_head",
            n_heads=model.config.num_attention_heads,
        )

        if model.config.architectures[0] == "LlamaForCausalLM":
            layer_idx = int(layer.split(".")[2])
        else:
            layer_idx = int(layer.split(".")[4])

        if sender_layer == layer_idx:
            for bi in range(batch_size):
                if rel_pos == -1:
                    # Computing the previous query box label token position
                    sender_head_pos_in_clean = compute_prev_query_box_pos(
                        base_tokens[bi], clean_last_token_indices[bi]
                    )

                    # Since, queery box is not present in the prompt, patch in
                    # the output of heads from any random box label token,
                    # i.e. `clean_prev_box_label_pos`
                    sender_head_pos_in_corrupt = random.choice(range(6, 49, 7))
                else:
                    # Computing the position from which clean patching should start
                    # If rel_pos = 0, then patching should start from the last token
                    # If rel_pos = 2, then patching should start from the third last token
                    sender_head_pos_in_clean = clean_last_token_indices[bi] - rel_pos

                    # Computing the position from which the output of the sender
                    # head should be patched
                    sender_head_pos_in_corrupt = (
                        corrupt_last_token_indices[bi] - rel_pos
                    )

                # Patch clean output to all the heads of this layer from the
                # `sender_head_pos_in_clean` position to last token position,
                # except the sender head which is patched from the
                # `sender_head_pos_in_corrupt` position of the corrupt output
                for pos in range(
                    sender_head_pos_in_clean, clean_last_token_indices[bi] + 1
                ):
                    for head_idx in range(model.config.num_attention_heads):
                        if head_idx == sender_head and pos == sender_head_pos_in_clean:
                            input[bi, pos, sender_head] = corrupt_head_outputs[
                                bi, sender_head_pos_in_corrupt, sender_head
                            ]
                        else:
                            input[bi, pos, head_idx] = clean_head_outputs[
                                bi, pos, head_idx
                            ]

        else:
            for bi in range(batch_size):
                if rel_pos == -1:
                    # Computing the previous query box label token position
                    sender_head_pos_in_clean = compute_prev_query_box_pos(
                        base_tokens[bi], clean_last_token_indices[bi]
                    )
                else:
                    # Computing the position from which clean patching should start
                    # If rel_pos = 0, then patching should start from the last token
                    # If rel_pos = 2, then patching should start from the third last token (query box label token)
                    sender_head_pos_in_clean = clean_last_token_indices[bi] - rel_pos

                # Patch clean output to all the heads of this layer from the
                # `sender_head_pos_in_clean` position to last token position
                # since none of them are sender heads
                for pos in range(
                    sender_head_pos_in_clean, clean_last_token_indices[bi] + 1
                ):
                    input[bi, pos] = clean_head_outputs[bi, pos]

        input = rearrange(
            input,
            "batch seq_len n_heads d_head -> batch seq_len (n_heads d_head)",
            n_heads=model.config.num_attention_heads,
        )

        w_o = model.state_dict()[f"{layer}.weight"]
        output = einsum(
            input,
            w_o,
            "batch seq_len hidden_size, d_model hidden_size -> batch seq_len d_model",
        )

    return output


def patching_receiver_heads(
    output=None,
    layer=None,
    model: LlamaForCausalLM = None,
    base_tokens: list = None,
    patched_cache: dict = None,
    receiver_heads: list = None,
    clean_last_token_indices: list = None,
    rel_pos: int = None,
    batch_size: int = None,
):
    """
    Patches the input of the receiver head, i.e. key, query or value vectors.

    Args:
        output: output of the layer.
        layer: layer to patch.
        model: model to patch.
        base_tokens: clean inputs.
        patched_cache: patched cache of the model.
        receiver_heads: receiver heads.
        clean_last_token_indices: clean last token indices.
        rel_pos: relative position of the query box label token.
        batch_size: batch size of the dataloader.
    """

    if model.config.architectures[0] == "LlamaForCausalLM":
        receiver_heads_in_curr_layer = [
            h for l, h in receiver_heads if l == int(layer.split(".")[2])
        ]
    else:
        receiver_heads_in_curr_layer = [
            h for l, h in receiver_heads if l == int(layer.split(".")[4])
        ]

    # `output` is the output of k_proj, q_proj, or v_proj, i.e. input of
    # the receiver head
    output = rearrange(
        output,
        "batch seq_len (n_heads d_head) -> batch seq_len n_heads d_head",
        n_heads=model.config.num_attention_heads,
    )
    patched_head_outputs = rearrange(
        patched_cache[layer].output,
        "batch seq_len (n_heads d_head) -> batch seq_len n_heads d_head",
        n_heads=model.config.num_attention_heads,
    )

    for receiver_head in receiver_heads_in_curr_layer:
        for bi in range(batch_size):
            if rel_pos == -1:
                receiver_head_pos_in_clean = compute_prev_query_box_pos(
                    base_tokens[bi], clean_last_token_indices[bi]
                )
            else:
                receiver_head_pos_in_clean = clean_last_token_indices[bi] - rel_pos

            # Patch the input of receiver heads (output of k_proj, q_proj, or v_proj)
            # computed in the previous step of path patching
            output[bi, receiver_head_pos_in_clean, receiver_head] = (
                patched_head_outputs[bi, receiver_head_pos_in_clean, receiver_head]
            )

    output = rearrange(
        output,
        "batch seq_len n_heads d_head -> batch seq_len (n_heads d_head)",
        n_heads=model.config.num_attention_heads,
    )

    return output


def get_receiver_layers(
    model: LlamaForCausalLM, receiver_heads: list, composition: str
):
    """
    Gets the receiver layers from the receiver heads.

    Args:
        model: model under invetigation.
        receiver_heads: receiver heads.
        composition: composition to use for the receiver heads (k/q/v).
    """

    if model.config.architectures[0] == "LlamaForCausalLM":
        receiver_layers = list(
            set(
                [
                    f"model.layers.{layer}.self_attn.{composition}_proj"
                    for layer, _ in receiver_heads
                ]
            )
        )
    else:
        receiver_layers = list(
            set(
                [
                    f"base_model.model.model.layers.{layer}.self_attn.{composition}_proj"
                    for layer, _ in receiver_heads
                ]
            )
        )

    return receiver_layers


def load_eval_data(
    tokenizer: LlamaTokenizer, datafile: str, num_samples: int, batch_size: int
):
    """
    Loads the dataset for evaluation.

    Args:
        tokenizer: tokenizer to use.
        datafile: path to the datafile.
        num_samples: number of samples to use from the datafile.
        batch_size: batch size to use for the dataloader.
    """

    print("Loading dataset...")
    raw_data = sample_box_data(
        tokenizer=tokenizer,
        num_samples=num_samples,
        data_file=datafile,
    )
    dataset = Dataset.from_dict(
        {
            "input_ids": raw_data[0],
            "last_token_indices": raw_data[1],
            "labels": raw_data[2],
        }
    ).with_format("torch")
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False
    )

    return data_loader


def load_ablation_data(
    tokenizer: LlamaTokenizer,
    datafile: str,
    num_samples: int,
    batch_size: int,
    num_boxes: int = 7,
):
    """
    Loads the dataset for ablation.

    Args:
        tokenizer: tokenizer to use.
        datafile: path to the datafile.
        num_samples: number of samples to use from the datafile.
        batch_size: batch size to use for the dataloader.
        num_boxes: number of boxes in the datafile.
    """

    raw_data = get_data_for_mean_ablation(
        tokenizer=tokenizer,
        num_samples=3500,
        data_file=datafile,
        num_boxes=7,
    )

    ablate_dataset = Dataset.from_dict(
        {
            "input_ids": raw_data[0],
            "last_token_indices": raw_data[1],
        }
    ).with_format("torch")

    ablate_dataloader = torch.utils.data.DataLoader(
        ablate_dataset, batch_size=batch_size
    )
    return ablate_dataloader


def get_mean_activations(
    model: LlamaForCausalLM,
    tokenizer: LlamaTokenizer,
    datafile: str,
    num_samples: int,
    batch_size: int,
):
    """
    Computes the mean activations of every attention head at all positions.

    Args:
        model: model under investigation.
        tokenizer: tokenizer to use.
        datafile: path to the datafile.
        num_samples: number of samples to use from the datafile.
        batch_size: batch size to use for the dataloader.
    """

    print("Computing mean activations...")
    ablation_dataloader = load_ablation_data(
        tokenizer=tokenizer,
        datafile=datafile,
        num_samples=num_samples,
        batch_size=batch_size,
    )

    if model.config.architectures[0] == "LlamaForCausalLM":
        modules = [
            f"model.layers.{layer}.self_attn.o_proj"
            for layer in range(model.config.num_hidden_layers)
        ]
    else:
        modules = [
            f"base_model.model.model.layers.{layer}.self_attn.o_proj"
            for layer in range(model.config.num_hidden_layers)
        ]

    mean_activations = {}
    with torch.no_grad():
        for _, inp in enumerate(tqdm(ablation_dataloader)):
            for k, v in inp.items():
                if v is not None and isinstance(v, torch.Tensor):
                    inp[k] = v.to(model.device)

            with TraceDict(model, modules, retain_input=True) as cache:
                _ = model(inp["input_ids"])

            for layer in modules:
                if "o_proj" in layer:
                    if layer in mean_activations:
                        mean_activations[layer] += torch.sum(cache[layer].input, dim=0)
                    else:
                        mean_activations[layer] = torch.sum(cache[layer].input, dim=0)

            del cache
            torch.cuda.empty_cache()

        for layer in modules:
            mean_activations[layer] /= len(ablation_dataloader.dataset)

    return mean_activations, modules


def mean_ablate(
    inputs=None,
    output=None,
    layer=None,
    model: LlamaForCausalLM = None,
    circuit_components: dict = None,
    mean_activations: dict = None,
    input_tokens: torch.tensor = None,
    ablate_non_vital_pos: bool = None,
):
    """
    Ablates the model components that are not present in `circuit_components`
    by substituting their output with their corresponding mean activations.

    Args:
        inputs: inputs to the layer.
        output: output of the layer.
        layer: layer to patch.
        model: model to patch.
        circuit_components: circuit components.
        mean_activations: mean activations of the model.
        input_tokens: input tokens.
    """

    if isinstance(inputs, tuple):
        inputs = inputs[0]

    inputs = rearrange(
        inputs,
        "batch seq_len (n_heads d_head) -> batch seq_len n_heads d_head",
        n_heads=model.config.num_attention_heads,
    )

    mean_act = rearrange(
        mean_activations[layer],
        "seq_len (n_heads d_head) -> seq_len n_heads d_head",
        n_heads=model.config.num_attention_heads,
    )

    last_pos = inputs.size(1) - 1
    for bi in range(inputs.size(0)):
        prev_query_box_pos = compute_prev_query_box_pos(
            input_tokens[bi], input_tokens[bi].size(0) - 1
        )
        for token_pos in range(inputs.size(1)):
            if (
                token_pos != prev_query_box_pos
                and token_pos != last_pos - 2
                and token_pos != last_pos
                and ablate_non_vital_pos
            ):
                inputs[bi, token_pos, :] = mean_act[token_pos, :]

            elif token_pos == prev_query_box_pos:
                for head_idx in range(model.config.num_attention_heads):
                    if head_idx not in circuit_components[-1][layer]:
                        inputs[bi, token_pos, head_idx] = mean_act[token_pos, head_idx]

            elif token_pos == last_pos - 2:
                for head_idx in range(model.config.num_attention_heads):
                    if head_idx not in circuit_components[2][layer]:
                        inputs[bi, token_pos, head_idx] = mean_act[token_pos, head_idx]

            elif token_pos == last_pos:
                for head_idx in range(model.config.num_attention_heads):
                    if head_idx not in circuit_components[0][layer]:
                        inputs[bi, token_pos, head_idx] = mean_act[token_pos, head_idx]

    inputs = rearrange(
        inputs,
        "batch seq_len n_heads d_head -> batch seq_len (n_heads d_head)",
        n_heads=model.config.num_attention_heads,
    )
    w_o = model.state_dict()[f"{layer}.weight"]
    output = einsum(
        inputs,
        w_o,
        "batch seq_len hidden_size, d_model hidden_size -> batch seq_len d_model",
    )

    return output


def eval_circuit_performance(
    model: LlamaForCausalLM,
    dataloader: torch.utils.data.DataLoader,
    modules: list,
    circuit_components: dict,
    mean_activations: dict,
    ablate_non_vital_pos: bool = True,
):
    """
    Evaluates the performance of the model/circuit.

    Args:
        model: model under investigation.
        dataloader: dataloader containing clean and corrupt inputs.
        modules: modules to patch.
        circuit_components: circuit components.
        mean_activations: mean activations of the model.
    """

    correct_count, total_count = 0, 0
    with torch.no_grad():
        for _, inp in enumerate(tqdm(dataloader)):
            for k, v in inp.items():
                if v is not None and isinstance(v, torch.Tensor):
                    inp[k] = v.to(model.device)

            with TraceDict(
                model,
                modules,
                retain_input=True,
                edit_output=partial(
                    mean_ablate,
                    model=model,
                    circuit_components=circuit_components,
                    mean_activations=mean_activations,
                    input_tokens=inp["input_ids"],
                    ablate_non_vital_pos=ablate_non_vital_pos,
                ),
            ) as _:
                outputs = model(inp["input_ids"])

            for bi in range(inp["labels"].size(0)):
                label = inp["labels"][bi]
                pred = torch.argmax(outputs.logits[bi][inp["last_token_indices"][bi]])

                if label == pred:
                    correct_count += 1
                total_count += 1

            del outputs
            torch.cuda.empty_cache()

    current_acc = round(correct_count / total_count, 2)
    return current_acc


def get_circuit(
    model: LlamaForCausalLM,
    circuit_root_path: str,
    n_value_fetcher: int,
    n_pos_trans: int,
    n_pos_detect: int,
    n_struct_read: int,
):
    """
    Computes the circuit components.

    Args:
        model: model under investigation.
        circuit_root_path: path to the circuit components.
        n_value_fetcher: number of value fetcher heads.
        n_pos_trans: number of position transformer heads.
        n_pos_detect: number of position detector heads.
        n_struct_read: number of structure reader heads.
    """

    circuit_components = {}
    circuit_components[0] = defaultdict(list)
    circuit_components[2] = defaultdict(list)
    circuit_components[-1] = defaultdict(list)

    path = circuit_root_path + "/value_fetcher.pt"
    value_fetcher_heads = compute_topk_components(
        torch.load(path), k=n_value_fetcher, largest=False
    )

    path = circuit_root_path + "/pos_transmitter.pt"
    pos_transmitter_heads = compute_topk_components(
        torch.load(path), k=n_pos_trans, largest=False
    )

    path = circuit_root_path + "/pos_detector.pt"
    pos_detector_heads = compute_topk_components(
        torch.load(path), k=n_pos_detect, largest=False
    )

    path = circuit_root_path + "/struct_reader.pt"
    struct_reader_heads = compute_topk_components(
        torch.load(path), k=n_struct_read, largest=False
    )

    intersection = []
    for head in value_fetcher_heads:
        if head in pos_transmitter_heads:
            intersection.append(head)

    for head in intersection:
        value_fetcher_heads.remove(head)

    for layer_idx, head in value_fetcher_heads:
        if model.config.architectures[0] == "LlamaForCausalLM":
            layer = f"model.layers.{layer_idx}.self_attn.o_proj"
        else:
            layer = f"base_model.model.model.layers.{layer_idx}.self_attn.o_proj"
        circuit_components[0][layer].append(head)

    for layer_idx, head in pos_transmitter_heads:
        if model.config.architectures[0] == "LlamaForCausalLM":
            layer = f"model.layers.{layer_idx}.self_attn.o_proj"
        else:
            layer = f"base_model.model.model.layers.{layer_idx}.self_attn.o_proj"
        circuit_components[0][layer].append(head)

    for layer_idx, head in pos_detector_heads:
        if model.config.architectures[0] == "LlamaForCausalLM":
            layer = f"model.layers.{layer_idx}.self_attn.o_proj"
        else:
            layer = f"base_model.model.model.layers.{layer_idx}.self_attn.o_proj"
        circuit_components[2][layer].append(head)

    for layer_idx, head in struct_reader_heads:
        if model.config.architectures[0] == "LlamaForCausalLM":
            layer = f"model.layers.{layer_idx}.self_attn.o_proj"
        else:
            layer = f"base_model.model.model.layers.{layer_idx}.self_attn.o_proj"
        circuit_components[-1][layer].append(head)

    return (
        circuit_components,
        value_fetcher_heads,
        pos_transmitter_heads,
        pos_detector_heads,
        struct_reader_heads,
    )


def get_random_circuit(
    model: LlamaForCausalLM,
    circuit: dict,
):
    """
    Computes a random circuit with same #heads in each group as in the circuit.

    Args:
        model: model under investigation.
        n_value_fetcher: number of value fetcher heads.
        n_pos_trans: number of position transformer heads.
        n_pos_detect: number of position detector heads.
        n_struct_read: number of structure reader heads.
    """

    random_circuit = {}
    random_circuit[0] = defaultdict(list)
    random_circuit[2] = defaultdict(list)
    random_circuit[-1] = defaultdict(list)

    num_heads = model.config.num_attention_heads
    num_layers = model.config.num_hidden_layers
    n_value_fetcher = len(circuit["value_fetcher"])
    n_pos_transmitter = len(circuit["pos_transmitter"])
    n_pos_detector = len(circuit["pos_detector"])
    n_struct_reader = len(circuit["struct_reader"])

    heads_at_last_pos = np.random.choice(
        list(range(num_heads * num_layers)), n_value_fetcher + n_pos_transmitter
    )
    heads_at_query_box_pos = np.random.choice(
        list(range(num_heads * num_layers)), n_pos_detector
    )
    heads_at_prev_query_box_pos = np.random.choice(
        list(range(num_heads * num_layers)), n_struct_reader
    )

    heads_at_last_pos = [
        [head // num_layers, head % num_heads] for head in heads_at_last_pos
    ]
    heads_at_query_box_pos = [
        [head // num_layers, head % num_heads] for head in heads_at_query_box_pos
    ]
    heads_at_prev_query_box_pos = [
        [head // num_layers, head % num_heads] for head in heads_at_prev_query_box_pos
    ]

    for layer_idx, head in heads_at_last_pos:
        if model.config.architectures[0] == "LlamaForCausalLM":
            layer = f"model.layers.{layer_idx}.self_attn.o_proj"
        else:
            layer = f"base_model.model.model.layers.{layer_idx}.self_attn.o_proj"
        random_circuit[0][layer].append(head)

    for layer_idx, head in heads_at_query_box_pos:
        if model.config.architectures[0] == "LlamaForCausalLM":
            layer = f"model.layers.{layer_idx}.self_attn.o_proj"
        else:
            layer = f"base_model.model.model.layers.{layer_idx}.self_attn.o_proj"
        random_circuit[2][layer].append(head)

    for layer_idx, head in heads_at_prev_query_box_pos:
        if model.config.architectures[0] == "LlamaForCausalLM":
            layer = f"model.layers.{layer_idx}.self_attn.o_proj"
        else:
            layer = f"base_model.model.model.layers.{layer_idx}.self_attn.o_proj"
        random_circuit[-1][layer].append(head)

    return random_circuit


def compute_pair_drop_values(
    model: LlamaForCausalLM,
    heads: list,
    circuit_components: dict,
    dataloader: torch.utils.data.DataLoader,
    modules: list,
    mean_activations: dict,
    rel_pos: int = 0,
):
    """
    Computes the pair drop values for the given heads.

    Args:
        model: model under investigation.
        heads: heads to compute the pair drop values for.
        circuit_components: circuit components.
        dataloader: dataloader containing clean and corrupt inputs.
        modules: modules to patch.
        mean_activations: mean activations of the model.
        rel_pos: relative position of the query box label token.
    """

    greedy_res = defaultdict(lambda: defaultdict(float))

    for layer_idx_1, head_1 in tqdm(heads, total=len(heads), desc="Pair drop values"):
        if model.config.architectures[0] == "LlamaForCausalLM":
            layer_1 = f"model.layers.{layer_idx_1}.self_attn.o_proj"
        else:
            layer_1 = f"base_model.model.model.layers.{layer_idx_1}.self_attn.o_proj"

        circuit_components[rel_pos][layer_1].remove(head_1)

        for layer_idx_2, head_2 in heads:
            if model.config.architectures[0] == "LlamaForCausalLM":
                layer_2 = f"model.layers.{layer_idx_2}.self_attn.o_proj"
            else:
                layer_2 = (
                    f"base_model.model.model.layers.{layer_idx_2}.self_attn.o_proj"
                )

            if greedy_res[(layer_2, head_2)][(layer_1, head_1)] > 0.0:
                continue
            if layer_1 is not layer_2 and head_1 is not head_2:
                circuit_components[rel_pos][layer_2].remove(head_2)

            greedy_res[(layer_1, head_1)][(layer_2, head_2)] = eval_circuit_performance(
                model, dataloader, modules, circuit_components, mean_activations
            )
            if layer_1 is not layer_2 and head_1 is not head_2:
                circuit_components[rel_pos][layer_2].append(head_2)

        circuit_components[rel_pos][layer_1].append(head_1)

    res = defaultdict(lambda: defaultdict(float))
    for k in greedy_res:
        for k_2 in greedy_res[k]:
            if greedy_res[k][k_2] > 0.0:
                res[str(k)][str(k_2)] = greedy_res[k][k_2]
                res[str(k_2)][str(k)] = greedy_res[k][k_2]

    return res


def get_head_significance_score(
    model: LlamaForCausalLM,
    heads: list,
    ranked: dict,
    percentage: float,
    circuit_components: dict,
    dataloader: torch.utils.data.DataLoader,
    modules: list,
    mean_activations: dict,
    rel_pos: int,
):
    """
    Computes the head significance score for the given heads.

    Args:
        model: model under investigation.
        heads: heads to compute the pair drop values for.
        ranked: ranked pair drop values.
        percentage: percentage of heads to use for computing the head significance score.
        circuit_components: circuit components.
        dataloader: dataloader containing clean and corrupt inputs.
        modules: modules to patch.
        mean_activations: mean activations of the model.
        rel_pos: relative position of the query box label token.
    """

    res = {}

    for layer_idx, head in tqdm(
        heads, total=len(heads), desc="Head significance score"
    ):
        if model.config.architectures[0] == "LlamaForCausalLM":
            layer = f"model.layers.{layer_idx}.self_attn.o_proj"
        else:
            layer = f"base_model.model.model.layers.{layer_idx}.self_attn.o_proj"

        for r in ranked[str((layer, head))][
            : math.ceil(percentage * len(ranked.values()))
        ]:
            top_layer = r[0].split(",")[0][2:-1]
            top_head = int(r[0].split(",")[1][:-1])
            if r[1] <= 0:
                break
            circuit_components[rel_pos][top_layer].remove(top_head)

        before = eval_circuit_performance(
            model, dataloader, modules, circuit_components, mean_activations
        )
        circuit_components[rel_pos][layer].remove(head)
        after = eval_circuit_performance(
            model, dataloader, modules, circuit_components, mean_activations
        )
        res[(layer, head)] = (before, after)

        for r in ranked[str((layer, head))][
            : math.ceil(percentage * len(ranked.values()))
        ]:
            top_layer = r[0].split(",")[0][2:-1]
            top_head = int(r[0].split(",")[1][:-1])
            if r[1] <= 0:
                break
            circuit_components[rel_pos][top_layer].append(top_head)
        circuit_components[rel_pos][layer].append(head)

    return res


def get_final_circuit(model, circuit_heads):
    """
    Computes the final circuit.

    Args:
        model: model under investigation.
        circuit_heads: circuit heads.
    """

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
