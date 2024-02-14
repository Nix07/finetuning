import os
import json
import sys
import torch

from transformers import LlamaTokenizer, LlamaForCausalLM
from baukit import TraceDict
from einops import einsum, rearrange
from tqdm import tqdm
from functools import partial
from collections import defaultdict
from datasets import Dataset
from torch.utils.data import DataLoader
from peft import PeftModel
from typing import Callable

curr_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.abspath(os.path.join(curr_dir, os.pardir))
sys.path.append(parent_dir)
from data.data_utils import *

torch.manual_seed(20)


def get_model_and_tokenizer(model_name: str, device: str = "cuda"):
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
        path = "/data/nikhil_prakash/llama_weights/7B/"
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


def create_dataloaders(raw_data: dict = None, batch_size: int = 32):
    """
    Loads the data into dataloaders.

    Args:
        raw_data (tuple): Tuple of lists containing the raw data.
        batch_size (int): Batch size for the dataloaders.
    """

    train_size = int(0.5 * len(raw_data[0]))
    eval_size = int(0.25 * len(raw_data[0]))

    print("Train size: ", train_size)
    print("Eval size: ", eval_size)
    print("Test size: ", len(raw_data[0]) - train_size - eval_size)

    raw_train = (
        raw_data[0][:train_size],
        raw_data[1][:train_size],
        raw_data[2][:train_size],
        raw_data[3][:train_size],
        raw_data[4][:train_size],
    )
    raw_eval = (
        raw_data[0][train_size : train_size + eval_size],
        raw_data[1][train_size : train_size + eval_size],
        raw_data[2][train_size : train_size + eval_size],
        raw_data[3][train_size : train_size + eval_size],
        raw_data[4][train_size : train_size + eval_size],
    )
    raw_test = (
        raw_data[0][train_size + eval_size :],
        raw_data[1][train_size + eval_size :],
        raw_data[2][train_size + eval_size :],
        raw_data[3][train_size + eval_size :],
        raw_data[4][train_size + eval_size :],
    )

    train_dataset = Dataset.from_dict(
        {
            "base_input_ids": raw_train[0],
            "base_input_last_pos": raw_train[1],
            "source_input_ids": raw_train[2],
            "source_input_last_pos": raw_train[3],
            "labels": raw_train[4],
        }
    ).with_format("torch")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
    )

    eval_dataset = Dataset.from_dict(
        {
            "base_input_ids": raw_eval[0],
            "base_input_last_pos": raw_eval[1],
            "source_input_ids": raw_eval[2],
            "source_input_last_pos": raw_eval[3],
            "labels": raw_eval[4],
        }
    ).with_format("torch")
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
    )

    test_dataset = Dataset.from_dict(
        {
            "base_input_ids": raw_test[0],
            "base_input_last_pos": raw_test[1],
            "source_input_ids": raw_test[2],
            "source_input_last_pos": raw_test[3],
            "labels": raw_test[4],
        }
    ).with_format("torch")
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
    )

    return train_dataloader, eval_dataloader, test_dataloader


def load_activations(
    model: LlamaForCausalLM, modules: list, desiderata: DataLoader, device: str
):
    """
    Loads the activations of counterfactual examples to patch during the
    training process.

    Args:
        model (LlamaForCausalLM): Model to use.
        modules (list): List of modules to extract activations from.
        desiderata (DataLoader): Dataloader to use.
        device (str): Device to use.
    """

    from_activations_train = {}

    for di, desid in enumerate(desiderata):
        from_activations_train[di] = {}

        for bi, inputs in enumerate(desid):
            for k, v in inputs.items():
                if v is not None and isinstance(v, torch.Tensor):
                    inputs[k] = v.to(device)

            from_activations_train[di][bi] = {}
            with torch.no_grad():
                with TraceDict(model, modules, retain_input=True) as trace:
                    _ = model(inputs["source_input_ids"])

                    for module in modules:
                        if "self_attn" in module:
                            from_activations_train[di][bi][module] = (
                                trace[module].input.detach().cpu()
                            )
                        else:
                            from_activations_train[di][bi][module] = (
                                trace[module].output.detach().cpu()
                            )

            for k, v in inputs.items():
                if v is not None and isinstance(v, torch.Tensor):
                    inputs[k] = v.to("cpu")

            del trace
            torch.cuda.empty_cache()

    return from_activations_train


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


def edit_output(
    inputs: tuple = None,
    output: torch.Tensor = None,
    layer: str = None,
    model: LlamaForCausalLM = None,
    mask: torch.Tensor = None,
    from_activations: dict = None,
    to_last_token_pos: torch.Tensor = None,
    from_last_token_pos: torch.Tensor = None,
    rel_pos: int = None,
    input_tokens: dict = None,
    device: str = None,
    mask_dict: dict = None,
):
    """
    Edits the output of the model by patching from counterfactual examples.

    Args:
        inputs (tuple): Tuple containing the inputs.
        output (torch.Tensor): Output of the model.
        layer (str): Layer to patch.
        model (LlamaForCausalLM): Model to use.
        mask (torch.Tensor): Mask to use.
        from_activations (dict): Dictionary containing the activations.
        to_last_token_pos (torch.Tensor): Last token position of the target.
        from_last_token_pos (torch.Tensor): Last token position of the source.
        rel_pos (int): Relative position of the token to patch.
        input_tokens (dict): Dictionary containing the input tokens.
        device (str): Device to use.
        mask_dict (dict): Dictionary containing the mask.
    """
    num_heads = model.config.num_attention_heads
    head_size = model.config.hidden_size // num_heads

    if "self_attn" in layer:
        inp = inputs[0]
        from_activations[layer] = from_activations[layer].to(device)

        # Computing the output of each head in this layer after the intervention
        for head_idx in range(num_heads):
            head_start = head_idx * head_size
            head_end = (head_idx + 1) * head_size

            if f"{layer}.{head_idx}" in mask_dict:
                abl_amt = mask[mask_dict[f"{layer}.{head_idx}"]]

                for batch in range(inp.shape[0]):
                    if rel_pos != -1:
                        intervention = (
                            abl_amt
                            * inp[
                                batch,
                                to_last_token_pos[batch] - rel_pos,
                                head_start:head_end,
                            ].clone()
                            + (1 - abl_amt)
                            * from_activations[layer][
                                batch,
                                from_last_token_pos[batch] - rel_pos,
                                head_start:head_end,
                            ]
                        )
                        inp[
                            batch,
                            to_last_token_pos[batch] - rel_pos,
                            head_start:head_end,
                        ] = intervention
                    else:
                        base_prev_box_token_pos = compute_prev_query_box_pos(
                            input_tokens["base_input_ids"][batch],
                            input_tokens["base_input_last_pos"][batch],
                        )
                        source_prev_box_token_pos = compute_prev_query_box_pos(
                            input_tokens["source_input_ids"][batch],
                            input_tokens["source_input_last_pos"][batch],
                        )

                        intervention = (
                            abl_amt
                            * inp[
                                batch, base_prev_box_token_pos, head_start:head_end
                            ].clone()
                            + (1 - abl_amt)
                            * from_activations[layer][
                                batch, source_prev_box_token_pos, head_start:head_end
                            ]
                        )
                        inp[batch, base_prev_box_token_pos, head_start:head_end] = (
                            intervention
                        )

        from_activations[layer] = from_activations[layer].to("cpu")

        weights = model.state_dict()[f"{layer}.weight"]
        mod_output = einsum(
            inp,
            weights,
            "batch seq_len hidden_size, d_model hidden_size -> batch seq_len d_model",
        )

        del weights
        torch.cuda.empty_cache()
        return mod_output

    else:
        assert False, "shouldn't be here"


def get_data(
    desid_method: Callable = None,
    tokenizer: LlamaTokenizer = None,
    data_file: str = None,
    object_file: str = None,
    batch_size: int = 32,
):
    """
    Loads the data into dataloaders.

    Args:
        desid_method (function): Function to use to load the data.
        tokenizer (LlamaTokenizer): Tokenizer to use.
        data_file (str): Path to the data file.
        object_file (str): Path to the object file.
        num_boxes (int): Number of boxes to use.
        batch_size (int): Batch size for the dataloaders.
    """
    raw_data = desid_method(
        tokenizer=tokenizer,
        num_samples=2000,
        data_file=data_file,
        object_file=object_file,
        num_boxes=7,
        alt_format=True,
        correct_pred_indices=[],
    )

    train, valid, test = create_dataloaders(raw_data, batch_size=batch_size)

    return [train], [valid], [test]


def get_circuit_components(model, circuit_path):
    """
    Loads the circuit components from the circuit path.

    Args:
        model (LlamaForCausalLM): Model to use.
        circuit_path (str): Path to the circuit file.
    """

    circuit_components = {}
    circuit_components[0] = defaultdict(list)
    circuit_components[2] = defaultdict(list)
    circuit_components[-1] = defaultdict(list)

    with open(circuit_path, "r", encoding="utf-8") as f:
        circuit_heads = json.load(f)

    value_fetcher = circuit_heads["value_fetcher"]
    pos_transmitter = circuit_heads["pos_transmitter"]
    pos_detector = circuit_heads["pos_detector"]
    struct_reader = circuit_heads["struct_reader"]

    head_groups = {
        "value_fetcher": value_fetcher,
        "pos_transmitter": pos_transmitter,
        "pos_detector": pos_detector,
        "struct_reader": struct_reader,
    }

    print(f"Value Fetcher heads: {len(value_fetcher)}")
    print(f"Position Transmitter heads: {len(pos_transmitter)}")
    print(f"Position Detector Heads: {len(pos_detector)}")
    print(f"Structure Reader Heads: {len(struct_reader)}")

    for layer_idx, head in value_fetcher:
        if model.config.architectures[0] == "LlamaForCausalLM":
            layer = f"model.layers.{layer_idx}.self_attn.o_proj"
        else:
            layer = f"base_model.model.model.layers.{layer_idx}.self_attn.o_proj"
        circuit_components[0][layer].append(head)

    for layer_idx, head in pos_transmitter:
        if model.config.architectures[0] == "LlamaForCausalLM":
            layer = f"model.layers.{layer_idx}.self_attn.o_proj"
        else:
            layer = f"base_model.model.model.layers.{layer_idx}.self_attn.o_proj"
        circuit_components[0][layer].append(head)

    for layer_idx, head in pos_detector:
        if model.config.architectures[0] == "LlamaForCausalLM":
            layer = f"model.layers.{layer_idx}.self_attn.o_proj"
        else:
            layer = f"base_model.model.model.layers.{layer_idx}.self_attn.o_proj"
        circuit_components[2][layer].append(head)

    for layer_idx, head in struct_reader:
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

    return circuit_components, head_groups


def compute_heads_from_mask(
    model: LlamaForCausalLM, mask_dict: dict, rounded: torch.Tensor
):
    """
    Computes the heads from the mask.

    Args:
        model (LlamaForCausalLM): Model to use.
        mask_dict (dict): Dictionary containing the mask.
        rounded (torch.Tensor): Rounded mask.
    """

    masked_heads = []
    inverse_mask_dict = {v: k for k, v in mask_dict.items()}

    for mask_idx in (rounded == 0).nonzero()[:, 0]:
        layer = inverse_mask_dict[mask_idx.item()]
        if model.config.architectures[0] == "LlamaForCausalLM":
            layer_idx = int(layer.split(".")[2])
        else:
            layer_idx = int(layer.split(".")[4])

        head_idx = int(layer.split(".")[-1])
        masked_heads.append([layer_idx, head_idx])

    return masked_heads


def load_data_for_act_patching(raw_data: list, batch_size: int):
    """
    Loads the data into dataloaders.

    Args:
        raw_data (tuple): Tuple containing the raw data.
        batch_size (int): Batch size for the dataloaders.
    """

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


def activation_patching(
    inputs: tuple = None,
    output: torch.Tensor = None,
    layer: str = None,
    model: LlamaForCausalLM = None,
    source_cache: dict = None,
    patching_heads: dict = None,
    bi: int = None,
    input_tokens: dict = None,
):
    """
    Patches the activations from corrupt example to the original run.

    Args:
        inputs (tuple): Tuple containing the inputs.
        output (torch.Tensor): Output of the model.
        layer (str): Layer to patch.
        model (LlamaForCausalLM): Model to use.
        source_cache (dict): Dictionary containing the corrupt activations.
        patching_heads (dict): Dictionary containing the heads to patch.
        bi (int): Batch index.
        input_tokens (dict): Dictionary containing the input tokens.
    """

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
                prev_query_box_pos = compute_prev_query_box_pos(
                    input_tokens["base_input_ids"][batch],
                    input_tokens["base_input_last_pos"][batch],
                )
                for head in curr_layer_heads:
                    inputs[batch, prev_query_box_pos, head] = cache[
                        batch, prev_query_box_pos, head
                    ]
        else:
            b_pos = inputs.size(1) - rel_pos - 1
            s_pos = cache.size(1) - rel_pos - 1
            for head in curr_layer_heads:
                inputs[:, b_pos, head] = cache[:, s_pos, head]

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

    del w_o
    torch.cuda.empty_cache()
    return output
