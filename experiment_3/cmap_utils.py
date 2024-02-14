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
from data.data_utils import sample_box_data


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
        # path = "/home/local_nikhil/Projects/llama_weights/7B/"
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


def load_data(
    tokenizer: LlamaTokenizer,
    data_file: str,
    num_samples: int = 500,
    batch_size: int = 8,
):
    """
    Loads the data from the data file.

    Args:
        tokenizer (LlamaTokenizer): The tokenizer to use.
        data_file (str): Path to the data file.
        num_samples (int, optional): Number of samples to load. Defaults to 500.
        batch_size (int, optional): Batch size. Defaults to 8.
    """

    raw_data = sample_box_data(
        tokenizer=tokenizer,
        num_samples=num_samples,
        data_file=data_file,
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

    return dataloader


def eval_model_performance(
    model: LlamaForCausalLM, dataloader: DataLoader, device: str
):
    """
    Evaluates the model performance on the given dataloader.

    Args:
        model (LlamaForCausalLM): The model to evaluate.
        dataloader (DataLoader): The dataloader to use for evaluation.
    """
    total_count = 0
    correct_count = 0
    model.eval()
    with torch.no_grad():
        for _, output in tqdm(enumerate(tqdm(dataloader))):
            for k, v in output.items():
                if v is not None and isinstance(v, torch.Tensor):
                    output[k] = v.to(model.device)

            outputs = model(input_ids=output["input_ids"])

            for bi in range(output["labels"].size(0)):
                label = output["labels"][bi]
                pred = torch.argmax(
                    outputs.logits[bi][output["last_token_indices"][bi]]
                )

                if label == pred:
                    correct_count += 1
                total_count += 1

    del outputs
    torch.cuda.empty_cache()

    current_acc = round(correct_count / total_count, 2)
    return current_acc


def cmap_in(
    inputs: tuple = None,
    outputs: torch.Tensor = None,
    layer: str = None,
    model: LlamaForCausalLM = None,
    goat_cache: dict = None,
    llama_cache: dict = None,
    patching_component: list = None,
    bi: int = None,
    pos_heads_dict: dict = None,
    input_tokens: dict = None,
):
    """
    Patches the input, defined by `patching_component`, of attention heads defined
    in `pos_heads_dict` with its input in the Goat model.

    Args:
        inputs (tuple, optional): Inputs to the layer. Defaults to None.
        outputs (torch.Tensor, optional): Outputs of the layer. Defaults to None.
        layer (str, optional): Layer name. Defaults to None.
        model (LlamaForCausalLM, optional): The model to patch. Defaults to None.
        goat_cache (dict, optional): The cache of the goat model. Defaults to None.
        llama_cache (dict, optional): The cache of the llama model. Defaults to None.
        patching_component (list, optional): The patching component. Defaults to None.
        bi (int, optional): Batch index. Defaults to None.
        pos_heads_dict (dict, optional): Dictionary of relative positions and heads. Defaults to None.
        input_tokens (dict, optional): Dictionary of input tokens. Defaults to None.
    """

    if isinstance(inputs, tuple):
        inputs = inputs[0]

    g_cache = rearrange(
        goat_cache[bi][layer],
        "batch seq_len (n_heads d_head) -> batch seq_len n_heads d_head",
        n_heads=model.config.num_attention_heads,
    )

    l_cache = rearrange(
        llama_cache[bi][layer],
        "batch seq_len (n_heads d_head) -> batch seq_len n_heads d_head",
        n_heads=model.config.num_attention_heads,
    )

    # Since we are patching the value and key vectors of attention heads at all
    # positions, it is important to make sure the output of those affected heads
    # remain the same. Hence, we patch the output of the affected heads with their
    # output in the llama model.
    if "o_proj" in layer:
        inputs = rearrange(
            inputs,
            "batch seq_len (n_heads d_head) -> batch seq_len n_heads d_head",
            n_heads=model.config.num_attention_heads,
        )
        inputs[:, :-1, :] = l_cache[:, :-1, :]
        inputs = rearrange(
            inputs,
            "batch seq_len n_heads d_head -> batch seq_len (n_heads d_head)",
            n_heads=model.config.num_attention_heads,
        )
        w_o = model.state_dict()[f"{layer}.weight"]
        outputs = einsum(
            inputs,
            w_o,
            "batch seq_len hidden_size, d_model hidden_size -> batch seq_len d_model",
        )

    else:
        outputs = rearrange(
            outputs,
            "batch seq_len (n_heads d_head) -> batch seq_len n_heads d_head",
            n_heads=model.config.num_attention_heads,
        )

        for rel_pos, heads in pos_heads_dict.items():
            curr_layer_heads = [h for l, h in heads if l == int(layer.split(".")[2])]

            if rel_pos == -1:
                for batch in range(inputs.size(0)):
                    prev_query_box_pos = compute_prev_query_box_pos(
                        input_tokens["input_ids"][batch],
                        input_tokens["last_token_indices"][batch],
                    )
                    if "v_proj" in layer and "v_proj" in patching_component:
                        for head in curr_layer_heads:
                            outputs[:, :prev_query_box_pos, head] = g_cache[
                                :, :prev_query_box_pos, head
                            ]

                    if "k_proj" in layer and "k_proj" in patching_component:
                        for head in curr_layer_heads:
                            outputs[:, :prev_query_box_pos, head] = g_cache[
                                :, :prev_query_box_pos, head
                            ]

                    if "q_proj" in layer and "q_proj" in patching_component:
                        for head in curr_layer_heads:
                            outputs[:, prev_query_box_pos, head] = g_cache[
                                :, prev_query_box_pos, head
                            ]
            else:
                heads_pos = input_tokens["input_ids"].size(1) - rel_pos - 1
                if "v_proj" in layer and "v_proj" in patching_component:
                    for head_idx in curr_layer_heads:
                        outputs[:, :heads_pos, head_idx] = g_cache[
                            :, :heads_pos, head_idx
                        ]

                if "k_proj" in layer and "k_proj" in patching_component:
                    for head_idx in curr_layer_heads:
                        outputs[:, :heads_pos, head_idx] = g_cache[
                            :, :heads_pos, head_idx
                        ]

                if "q_proj" in layer and "q_proj" in patching_component:
                    for head_idx in curr_layer_heads:
                        outputs[:, heads_pos, head_idx] = g_cache[
                            :, heads_pos, head_idx
                        ]

        outputs = rearrange(
            outputs,
            "batch seq_len n_heads d_head -> batch seq_len (n_heads d_head)",
            n_heads=model.config.num_attention_heads,
        )

    return outputs


def cmap_out(
    inputs: tuple = None,
    outputs: torch.Tensor = None,
    layer: str = None,
    model: LlamaForCausalLM = None,
    finetuned_cache: dict = None,
    bi: int = None,
    pos_heads_dict: dict = None,
    input_tokens: dict = None,
):
    """
    Patches the output of attention heads defined in `pos_heads_dict` with its output
    in the Finetuned model.

    Args:
        inputs (tuple, optional): Inputs to the layer. Defaults to None.
        outputs (torch.Tensor, optional): Outputs of the layer. Defaults to None.
        layer (str, optional): Layer name. Defaults to None.
        model (LlamaForCausalLM, optional): The model to patch. Defaults to None.
        finetuned_cache (dict, optional): The cache of the fine-tuned model. Defaults to None.
        bi (int, optional): Batch index. Defaults to None.
        pos_heads_dict (dict, optional): Dictionary of relative positions and heads. Defaults to None.
        input_tokens (dict, optional): Dictionary of input tokens. Defaults to None.
    """

    if isinstance(inputs, tuple):
        inputs = inputs[0]

    if "o_proj" in layer:
        ft_cache = rearrange(
            finetuned_cache[bi][layer],
            "batch seq_len (n_heads d_head) -> batch seq_len n_heads d_head",
            n_heads=model.config.num_attention_heads,
        )

        inputs = rearrange(
            inputs,
            "batch seq_len (n_heads d_head) -> batch seq_len n_heads d_head",
            n_heads=model.config.num_attention_heads,
        )

        for rel_pos, heads in pos_heads_dict.items():
            curr_layer_heads = [h for l, h in heads if l == int(layer.split(".")[2])]

            if rel_pos == -1:
                for batch in range(inputs.size(0)):
                    prev_query_box_pos = compute_prev_query_box_pos(
                        input_tokens["input_ids"][batch],
                        input_tokens["last_token_indices"][batch],
                    )
                    for head in curr_layer_heads:
                        inputs[batch, prev_query_box_pos, head] = ft_cache[
                            batch, prev_query_box_pos, head
                        ]

            else:
                pos = input_tokens["input_ids"].size(1) - rel_pos - 1
                for head in curr_layer_heads:
                    inputs[:, pos, head] = ft_cache[:, pos, head]

        inputs = rearrange(
            inputs,
            "batch seq_len n_heads d_head -> batch seq_len (n_heads d_head)",
            n_heads=model.config.num_attention_heads,
        )
        w_o = model.state_dict()[f"{layer}.weight"]
        outputs = einsum(
            inputs,
            w_o,
            "batch seq_len hidden_size, d_model hidden_size -> batch seq_len d_model",
        )

    return outputs
