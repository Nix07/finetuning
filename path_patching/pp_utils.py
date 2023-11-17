import math
import sys
import torch
from transformers import (
    AutoModelForCausalLM,
    LlamaForCausalLM,
    LlamaTokenizer,
)
from baukit import TraceDict
from einops import rearrange, einsum
from peft import PeftModel
from datasets import Dataset
from functools import partial
from collections import defaultdict
from tqdm import tqdm

sys.path.append("/data/nikhil_prakash/anima-2.0/")
sys.path.append("../")
import analysis_utils
from counterfactual_datasets.entity_tracking import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_topk_components(patching_scores: torch.Tensor, k: int, largest=True):
    """Computes the topk most influential components (i.e. heads) for patching."""
    top_indices = torch.topk(patching_scores.flatten(), k, largest=largest).indices

    # Convert the top_indices to 2D indices
    row_indices = top_indices // patching_scores.shape[1]
    col_indices = top_indices % patching_scores.shape[1]
    top_components = torch.stack((row_indices, col_indices), dim=1)
    # Get the top indices as a list of 2D indices (row, column)
    top_components = top_components.tolist()
    return top_components


def load_model_tokenizer(model_name: str):
    print(f"Loading model...")
    tokenizer = LlamaTokenizer.from_pretrained(
        "hf-internal-testing/llama-tokenizer", padding_side="right"
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    if model_name == "llama":
        path = "../llama_7b/"
        model = AutoModelForCausalLM.from_pretrained(path).to(device)

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
        model = AutoModelForCausalLM.from_pretrained(path).to(device)

    return model, tokenizer


def load_pp_data(
    tokenizer: LlamaTokenizer, datafile: str, num_samples: int, num_boxes: int
):
    print(f"Loading dataset...")
    raw_data = box_index_aligner_examples(
        tokenizer,
        num_samples=num_samples,
        data_file=datafile,
        architecture="LLaMAForCausalLM",
        few_shot=False,
        alt_examples=True,
        num_ents_or_ops=num_boxes,
    )
    base_tokens = raw_data[0]
    base_last_token_indices = raw_data[1]
    source_tokens = raw_data[2]
    source_last_token_indices = raw_data[3]
    correct_answer_token = raw_data[4]

    return (
        base_tokens,
        base_last_token_indices,
        source_tokens,
        source_last_token_indices,
        correct_answer_token,
    )


def get_caches(model: AutoModelForCausalLM, base_tokens: list, source_tokens: list):
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

    with torch.no_grad():
        with TraceDict(
            model,
            hook_points,
            retain_input=True,
        ) as clean_cache:
            _ = model(base_tokens)

        with TraceDict(
            model,
            hook_points,
            retain_input=True,
        ) as corrupt_cache:
            _ = model(source_tokens)

    return clean_cache, corrupt_cache, hook_points


def patching_heads(
    inputs=None,
    output=None,
    layer: str = None,
    model: AutoModelForCausalLM = None,
    clean_cache: dict = None,
    corrupt_cache: dict = None,
    base_tokens: list = None,
    sender_layer: str = None,
    sender_head: str = None,
    clean_last_token_indices: list = None,
    corrupt_last_token_indices: list = None,
    rel_pos: int = None,
):
    """
    rel_pos: Represents the token position relative to the "real" (non-padded) last token in the sequence. All the heads at this position and subsequent positions need to patched from clean run, except the sender head at this position.
    """

    input = inputs[0]
    batch_size = input.size(0)

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
                    clean_prev_box_label_pos = (
                        analysis_utils.compute_prev_query_box_pos(
                            base_tokens[bi], clean_last_token_indices[bi]
                        )
                    )

                    # Since, queery box is not present in the prompt, patch in
                    # the output of heads from any random box label token, i.e. `clean_prev_box_label_pos`
                    corrupt_prev_box_label_pos = clean_prev_box_label_pos
                else:
                    clean_prev_box_label_pos = clean_last_token_indices[bi] - rel_pos
                    corrupt_prev_box_label_pos = (
                        corrupt_last_token_indices[bi] - rel_pos
                    )

                for pos in range(
                    clean_prev_box_label_pos, clean_last_token_indices[bi] + 1
                ):
                    for head_ind in range(model.config.num_attention_heads):
                        if head_ind == sender_head and pos == clean_prev_box_label_pos:
                            input[bi, pos, sender_head] = corrupt_head_outputs[
                                bi, corrupt_prev_box_label_pos, sender_head
                            ]
                        else:
                            input[bi, pos, head_ind] = clean_head_outputs[
                                bi, pos, head_ind
                            ]

        else:
            for bi in range(batch_size):
                if rel_pos == -1:
                    # Computing the previous query box label token position
                    clean_prev_box_label_pos = (
                        analysis_utils.compute_prev_query_box_pos(
                            base_tokens[bi], clean_last_token_indices[bi]
                        )
                    )
                else:
                    clean_prev_box_label_pos = clean_last_token_indices[bi] - rel_pos

                for pos in range(
                    clean_prev_box_label_pos, clean_last_token_indices[bi] + 1
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
    model: AutoModelForCausalLM = None,
    base_tokens: list = None,
    patched_cache: dict = None,
    receiver_heads: list = None,
    clean_last_token_indices: list = None,
    rel_pos: int = None,
):
    batch_size = output.size(0)
    if model.config.architectures[0] == "LlamaForCausalLM":
        receiver_heads_in_curr_layer = [
            h for l, h in receiver_heads if l == int(layer.split(".")[2])
        ]
    else:
        receiver_heads_in_curr_layer = [
            h for l, h in receiver_heads if l == int(layer.split(".")[4])
        ]

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

    # Patch in the output of the receiver heads from patched run
    for receiver_head in receiver_heads_in_curr_layer:
        for bi in range(batch_size):
            if rel_pos == -1:
                # Computing the previous query box label token position
                clean_prev_box_label_pos = analysis_utils.compute_prev_query_box_pos(
                    base_tokens[bi], clean_last_token_indices[bi]
                )
            else:
                clean_prev_box_label_pos = clean_last_token_indices[bi] - rel_pos

            output[bi, clean_prev_box_label_pos, receiver_head] = patched_head_outputs[
                bi, clean_prev_box_label_pos, receiver_head
            ]

    output = rearrange(
        output,
        "batch seq_len n_heads d_head -> batch seq_len (n_heads d_head)",
        n_heads=model.config.num_attention_heads,
    )

    return output


def get_receiver_layers(model: AutoModelForCausalLM, receiver_heads: list):
    if model.config.architectures[0] == "LlamaForCausalLM":
        receiver_layers = list(
            set(
                [
                    f"model.layers.{layer}.self_attn.v_proj"
                    for layer, _ in receiver_heads
                ]
            )
        )
    else:
        receiver_layers = list(
            set(
                [
                    f"base_model.model.model.layers.{layer}.self_attn.v_proj"
                    for layer, _ in receiver_heads
                ]
            )
        )

    return receiver_layers


def loal_eval_data(
    tokenizer: LlamaTokenizer, datafile: str, num_samples: int, batch_size: int
):
    print(f"Loading dataset...")
    raw_data = entity_tracking_example_sampler(
        tokenizer=tokenizer,
        num_samples=num_samples,
        data_file=datafile,
        few_shot=False,
        alt_examples=True,
        architecture="LlamaForCausalLM",
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


def load_ablate_data(
    tokenizer: LlamaTokenizer,
    datafile: str,
    num_samples: int,
    batch_size: int,
    num_boxes: int = 7,
):
    raw_data = generate_data_for_eval(
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
    model: AutoModelForCausalLM,
    tokenizer: LlamaTokenizer,
    datafile: str,
    num_samples: int,
    batch_size: int,
):
    print("Computing mean activations...")
    ablate_dataloader = load_ablate_data(
        tokenizer=tokenizer,
        datafile=datafile,
        num_samples=num_samples,
        batch_size=batch_size,
    )

    if model.config.architectures[0] == "LlamaForCausalLM":
        modules = [f"model.layers.{layer}.self_attn.o_proj" for layer in range(32)]
    else:
        modules = [
            f"base_model.model.model.layers.{layer}.self_attn.o_proj"
            for layer in range(32)
        ]

    mean_activations = {}
    with torch.no_grad():
        # Assuming a single batch
        for _, inp in enumerate(ablate_dataloader):
            for k, v in inp.items():
                if v is not None and isinstance(v, torch.Tensor):
                    inp[k] = v.to(model.device)

            with TraceDict(model, modules, retain_input=True) as cache:
                _ = model(inp["input_ids"])

            for layer in modules:
                if "self_attn" in layer:
                    if layer in mean_activations:
                        mean_activations[layer] += torch.mean(cache[layer].input, dim=0)
                    else:
                        mean_activations[layer] = torch.mean(cache[layer].input, dim=0)
                else:
                    if layer in mean_activations:
                        mean_activations[layer] += torch.mean(
                            cache[layer].output, dim=0
                        )
                    else:
                        mean_activations[layer] = torch.mean(cache[layer].output, dim=0)

            del cache
            torch.cuda.empty_cache()

        for layer in modules:
            mean_activations[layer] /= len(ablate_dataloader)

    return mean_activations, modules


def mean_ablate(
    inputs=None,
    output=None,
    layer=None,
    model: AutoModelForCausalLM = None,
    circuit_components: dict = None,
    mean_activations: dict = None,
    input_tokens: torch.tensor = None,
):
    if isinstance(inputs, tuple):
        inputs = inputs[0]

    if isinstance(output, tuple):
        output = output[0]

    inputs = rearrange(
        inputs,
        "batch seq_len (n_heads d_head) -> batch seq_len n_heads d_head",
        n_heads=model.config.num_attention_heads,
    )

    mean_act = rearrange(
        mean_activations[layer],
        "seq_len (n_heads d_head) -> 1 seq_len n_heads d_head",
        n_heads=model.config.num_attention_heads,
    )

    last_pos = inputs.size(1) - 1
    for bi in range(inputs.size(0)):
        prev_query_box_pos = analysis_utils.compute_prev_query_box_pos(
            input_tokens[bi], input_tokens[bi].size(0) - 1
        )
        for token_pos in range(inputs.size(1)):
            if (
                token_pos != prev_query_box_pos
                and token_pos != last_pos
                and token_pos != last_pos - 2
                and token_pos != prev_query_box_pos + 1
            ):
                inputs[bi, token_pos, :] = mean_act[0, token_pos, :]
            elif token_pos == prev_query_box_pos:
                for head_idx in range(model.config.num_attention_heads):
                    if head_idx not in circuit_components[-1][layer]:
                        inputs[bi, token_pos, head_idx] = mean_act[
                            0, token_pos, head_idx
                        ]
            elif token_pos == prev_query_box_pos + 1:
                for head_idx in range(model.config.num_attention_heads):
                    if head_idx not in circuit_components[-2][layer]:
                        inputs[bi, token_pos, head_idx] = mean_act[
                            0, token_pos, head_idx
                        ]
            elif token_pos == last_pos:
                for head_idx in range(model.config.num_attention_heads):
                    if head_idx not in circuit_components[0][layer]:
                        inputs[bi, token_pos, head_idx] = mean_act[
                            0, token_pos, head_idx
                        ]
            elif token_pos == last_pos - 2:
                for head_idx in range(model.config.num_attention_heads):
                    if head_idx not in circuit_components[2][layer]:
                        inputs[bi, token_pos, head_idx] = mean_act[
                            0, token_pos, head_idx
                        ]

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


def eval(
    model: AutoModelForCausalLM,
    dataloader: torch.utils.data.DataLoader,
    modules: list,
    circuit_components: dict,
    mean_activations: dict,
):
    correct_count, total_count = 0, 0
    with torch.no_grad():
        for _, inp in enumerate((dataloader)):
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
    # print(f"Task accuracy: {current_acc}")
    return current_acc


def get_circuit(
    model: AutoModelForCausalLM,
    circuit_root_path: str,
    n_value_fetcher: int,
    n_pos_trans: int,
    n_pos_detect: int,
    n_struct_read: int,
):
    circuit_components = {}
    circuit_components[0] = defaultdict(list)
    circuit_components[2] = defaultdict(list)
    circuit_components[-1] = defaultdict(list)
    circuit_components[-2] = defaultdict(list)

    path = circuit_root_path + "/direct_logit_heads.pt"
    logit_values = torch.load(path)
    direct_logit_heads = analysis_utils.compute_topk_components(
        torch.load(path), k=n_value_fetcher, largest=False
    )

    path = circuit_root_path + "/heads_affect_direct_logit.pt"
    logit_values = torch.load(path)
    heads_affecting_direct_logit_heads = analysis_utils.compute_topk_components(
        torch.load(path), k=n_pos_trans, largest=False
    )

    path = circuit_root_path + "/heads_at_query_box_pos.pt"
    logit_values = torch.load(path)
    head_at_query_box_token = analysis_utils.compute_topk_components(
        torch.load(path), k=n_pos_detect, largest=False
    )

    path = circuit_root_path + "/heads_at_prev_query_box_pos.pt"
    logit_values = torch.load(path)
    heads_at_prev_box_pos = analysis_utils.compute_topk_components(
        torch.load(path), k=n_struct_read, largest=False
    )

    intersection = []
    for head in direct_logit_heads:
        if head in heads_affecting_direct_logit_heads:
            intersection.append(head)

    for head in intersection:
        direct_logit_heads.remove(head)

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

    return (
        circuit_components,
        direct_logit_heads,
        heads_affecting_direct_logit_heads,
        head_at_query_box_token,
        heads_at_prev_box_pos,
    )


def compute_pair_drop_values(
    model: AutoModelForCausalLM,
    heads: list,
    circuit_components: dict,
    dataloader: torch.utils.data.DataLoader,
    modules: list,
    mean_activations: dict,
    rel_pos: int = 0,
):
    greedy_res = defaultdict(lambda: defaultdict(float))

    for layer_idx_1, head_1 in tqdm(heads):
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

            greedy_res[(layer_1, head_1)][(layer_2, head_2)] = eval(
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
    model: AutoModelForCausalLM,
    heads: list,
    ranked: dict,
    percentage: float,
    circuit_components: dict,
    dataloader: torch.utils.data.DataLoader,
    modules: list,
    mean_activations: dict,
    rel_pos: int,
):
    res = {}

    for layer_idx, head in tqdm(heads):
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

        befor = eval(model, dataloader, modules, circuit_components, mean_activations)
        circuit_components[rel_pos][layer].remove(head)
        after = eval(model, dataloader, modules, circuit_components, mean_activations)
        res[(layer, head)] = (befor, after)

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
