import math
import json
import torch
from einops import einsum, rearrange
from baukit import TraceDict
from functools import partial
from collections import defaultdict
from datasets import Dataset
from torch.utils.data import DataLoader
from data.data_utils import *

seed = 10
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


def apply_causal_mask(attn_scores):
    ignore = torch.tensor(torch.finfo(attn_scores.dtype).min)
    mask = torch.triu(
        torch.ones(attn_scores.size(-2), attn_scores.size(-1), device=attn_scores.device),
        diagonal=1,
    ).bool()
    attn_scores.masked_fill_(mask, ignore)
    return attn_scores


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def zero_ablation(inputs, output, layer, model, ablation_heads, last_token_pos):
    """Zeroes out the activations of the specified head in the specified layer."""
    input = inputs[0]
    batch_size = input.shape[0]
    layer_idx = int(layer.split(".")[2])

    if "o_proj" in layer:
        input = rearrange(
            input,
            "batch seq_len (n_heads d_head) -> batch seq_len n_heads d_head",
            n_heads=model.config.num_attention_heads,
        )

        ablation_heads_curr_layer = [h for l_idx, h in ablation_heads if l_idx == layer_idx]

        for head in ablation_heads_curr_layer:
            for bi in range(batch_size):
                input[bi, last_token_pos[bi], head, :] = 0

        input = rearrange(
            input,
            "batch seq_len n_heads d_head -> batch seq_len (n_heads d_head)",
            n_heads=model.config.num_attention_heads,
        )
        w_o = model.model.layers[layer_idx].self_attn.o_proj.weight
        output = einsum(
            input,
            w_o,
            "batch seq_len hidden_size, d_model hidden_size -> batch seq_len d_model",
        )

    return output


def get_attn_scores(model, tokens, layer, ablation_heads=None, last_token_pos=None):
    if model.config.architectures[0] == "LlamaForCausalLM":
        modules = [
            [
                f"model.layers.{i}.self_attn.k_proj",
                f"model.layers.{i}.self_attn.q_proj",
                f"model.layers.{i}.self_attn.v_proj",
                f"model.layers.{i}.self_attn.o_proj",
            ]
            for i in range(model.config.num_hidden_layers)
        ]
    else:
        modules = [
            [
                f"base_model.model.model.layers.{i}.self_attn.k_proj",
                f"base_model.model.model.layers.{i}.self_attn.q_proj",
                f"base_model.model.model.layers.{i}.self_attn.v_proj",
                f"base_model.model.model.layers.{i}.self_attn.o_proj",
            ]
            for i in range(model.config.num_hidden_layers)
        ]
    modules = [item for sublist in modules for item in sublist]

    with torch.no_grad():
        if ablation_heads is None:
            with TraceDict(model, modules) as residual:
                _ = model(tokens)
        else:
            with TraceDict(
                model,
                modules,
                retain_input=True,
                edit_output=partial(
                    zero_ablation,
                    ablation_heads=ablation_heads,
                    last_token_pos=last_token_pos,
                    model=model,
                ),
            ) as residual:
                _ = model(tokens)

    batch_size, seq_len = tokens.shape
    n_heads = model.config.num_attention_heads
    d_head = model.config.hidden_size // n_heads

    key = (
        residual[f"model.layers.{layer}.self_attn.k_proj"]
        .output.view(batch_size, seq_len, n_heads, d_head)
        .transpose(1, 2)
    )
    query = (
        residual[f"model.layers.{layer}.self_attn.q_proj"]
        .output.view(batch_size, seq_len, n_heads, d_head)
        .transpose(1, 2)
    )
    value = residual[f"model.layers.{layer}.self_attn.v_proj"].output.view(
        batch_size, seq_len, n_heads, d_head
    )

    kv_seq_len = key.shape[-2]
    cos, sin = model.model.layers[layer].self_attn.rotary_emb(value, seq_len=kv_seq_len)
    positions = [i for i in range(seq_len)]
    positions = torch.tensor(positions).unsqueeze(0).repeat(batch_size, 1).to("cuda")
    query, key = apply_rotary_pos_emb(query, key, cos, sin, positions)

    attn_scores = einsum(
        key,
        query,
        "batch n_heads key_pos d_head, batch n_heads query_pos d_head -> batch n_heads query_pos key_pos",
    )
    attn_scores = attn_scores / math.sqrt(d_head)
    attn_scores = apply_causal_mask(attn_scores)
    attn_scores = torch.softmax(attn_scores, dim=-1)

    return attn_scores, value


def perf_metric(
    patched_logits,
    answer,
    base_logits,
    source_logits,
    base_last_token_pos,
    source_last_token_pos,
):
    """Computes the impact of patching on the model's output logits on a scale of [0, 1]."""
    # TODO: Remove for loop
    score = 0
    patched = torch.log_softmax(
        patched_logits[0, base_last_token_pos],
        dim=-1,
    )
    corrupt = torch.log_softmax(
        source_logits[0, source_last_token_pos],
        dim=-1,
    )
    clean = torch.log_softmax(base_logits[0, -1], dim=-1)
    print(patched.shape, corrupt.shape, clean.shape)
    numerator = patched[answer] - corrupt[answer]
    denominator = clean[answer] - corrupt[answer]
    score += numerator / denominator

    return score / batch_size


def compute_topk_components(patching_scores: torch.Tensor, k: int, largest=True):
    """Computes the topk most influential components (i.e. heads) for patching."""
    top_indices = torch.topk(patching_scores.flatten(), k, largest=largest).indices
    top_values = torch.topk(patching_scores.flatten(), k, largest=largest).values

    # Convert the top_indices to 2D indices
    row_indices = top_indices // patching_scores.shape[1]
    col_indices = top_indices % patching_scores.shape[1]
    top_components = torch.stack((row_indices, col_indices), dim=1)
    # Get the top indices as a list of 2D indices (row, column)
    top_components = top_components.tolist()
    return top_components, top_values.tolist()


def comparison_metric(eval_preds, eval_labels, incorrect_objects):
    # eval_preds: (#batch, vocab_size)
    # eval_labels: (#batch)
    # incorrect_objects: (#batch, #incorrect_objects)
    total_count = 0
    correct_count = 0
    for pred, correct_object, incorrect_objs in zip(eval_preds, eval_labels, incorrect_objects):
        correctness = True
        for incorrect_object in incorrect_objs:
            if pred[correct_object] >= pred[incorrect_object]:
                continue
            else:
                correctness = False
                break

        if correctness:
            correct_count += 1

        total_count += 1

    accuracy = round(correct_count / total_count, 2)
    return {"accuracy": accuracy}


def top_token_comparison(eval_preds, eval_labels, incorrect_objects=None):
    # eval_preds: (#batch, vocab_size)
    # eval_labels: (#batch)
    total_count = 0
    correct_count = 0
    for eval_pred, eval_label in zip(eval_preds, eval_labels):
        pred_test_labels = torch.argmax(eval_pred, dim=-1)
        correct_count += 1 if eval_label == pred_test_labels else 0
        total_count += 1

    accuracy = round(correct_count / total_count, 2)
    return {"accuracy": accuracy}


def compute_prev_query_box_pos(input_ids, last_token_index):
    query_box_token = input_ids[last_token_index - 2]
    prev_query_box_token_pos = (
        (input_ids[: last_token_index - 2] == query_box_token).nonzero().item()
    )
    return prev_query_box_token_pos


def get_circuit_components(model, circuit_path):
    circuit_components = {}
    circuit_components[0] = defaultdict(list)
    circuit_components[2] = defaultdict(list)
    circuit_components[-1] = defaultdict(list)
    circuit_components[-2] = defaultdict(list)

    with open(circuit_path, "r") as f:
        circuit_heads = json.load(f)

    direct_logit_heads = circuit_heads["direct_logit_heads"]
    heads_affecting_direct_logit_heads = circuit_heads["heads_affecting_direct_logit_heads"]
    head_at_query_box_token = circuit_heads["head_at_query_box_token"]
    heads_at_prev_box_pos = circuit_heads["heads_at_prev_box_pos"]

    print(f"Direct logit heads: {len(direct_logit_heads)}")
    print(f"Heads affecting direct logit heads: {len(heads_affecting_direct_logit_heads)}")
    print(f"Heads at query box token: {len(head_at_query_box_token)}")
    print(f"Heads at prev query box token: {len(heads_at_prev_box_pos)}")

    head_groups = {
        "direct_logit_heads": direct_logit_heads,
        "heads_affect_direct_logit": heads_affecting_direct_logit_heads,
        "heads_at_query_box_pos": head_at_query_box_token,
        "heads_at_prev_query_box_pos": heads_at_prev_box_pos,
    }

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
            circuit_components[pos][layer_idx] = list(set(circuit_components[pos][layer_idx]))

    return circuit_components, head_groups


def load_activations(model, modules, desiderata, device):
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


def compute_heads_from_mask(model, mask_dict, rounded):
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


def get_ablation_data(tokenizer, data_file, batch_size):
    raw_data = get_data_for_mean_ablation(
        tokenizer=tokenizer,
        num_samples=3500,
        data_file=data_file,
        num_boxes=7,
    )

    ablate_dataset = Dataset.from_dict(
        {
            "input_ids": raw_data[0],
            "last_token_indices": raw_data[1],
        }
    ).with_format("torch")

    print(f"Length of dataset: {len(ablate_dataset)}")

    ablate_dataloader = DataLoader(ablate_dataset, batch_size=batch_size)
    return ablate_dataloader


def get_mean_activations(model, tokenizer, modules, data_file, batch_size):
    ablate_dataloader = get_ablation_data(tokenizer, data_file, batch_size)

    mean_activations = {}
    with torch.no_grad():
        for _, inp in enumerate(ablate_dataloader):
            for k, v in inp.items():
                if v is not None and isinstance(v, torch.Tensor):
                    inp[k] = v.to(model.device)

            with TraceDict(model, modules, retain_input=True) as cache:
                _ = model(inp["input_ids"])

            for layer in modules:
                if "o_proj" in layer:
                    if layer in mean_activations:
                        mean_activations[layer] += torch.mean(cache[layer].input, dim=0)
                    else:
                        mean_activations[layer] = torch.mean(cache[layer].input, dim=0)

            del cache
            torch.cuda.empty_cache()

        for layer in modules:
            mean_activations[layer] /= len(ablate_dataloader)

    return mean_activations


def mean_ablate(
    inputs=None,
    output=None,
    layer=None,
    model=None,
    circuit_components=None,
    mean_activations=None,
    input_tokens=None,
):
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

    seq_len = inputs.size(1) - 1
    for bi in range(inputs.size(0)):
        prev_query_box_pos = compute_prev_query_box_pos(
            input_tokens[bi], input_tokens[bi].size(0) - 1
        )
        for token_pos in range(seq_len):
            if (
                token_pos != seq_len
                and token_pos != seq_len - 2
                and token_pos != prev_query_box_pos
            ):
                inputs[bi, token_pos, :] = mean_act[token_pos, :]
            elif token_pos == seq_len:
                for head_idx in range(model.config.num_attention_heads):
                    if head_idx not in circuit_components[0][layer]:
                        inputs[bi, token_pos, head_idx] = mean_act[token_pos, head_idx]
            elif token_pos == seq_len - 2:
                for head_idx in range(model.config.num_attention_heads):
                    if head_idx not in circuit_components[2][layer]:
                        inputs[bi, token_pos, head_idx] = mean_act[token_pos, head_idx]
            elif token_pos == prev_query_box_pos:
                for head_idx in range(model.config.num_attention_heads):
                    if head_idx not in circuit_components[-1][layer]:
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


def eval(model, dataloader, modules, circuit_components, mean_activations):
    correct_count, total_count = 0, 0
    with torch.no_grad():
        for _, inp in enumerate(dataloader):
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
                # else:
                #     print(f"Label: {tokenizer.decode(label)}, Prediction: {tokenizer.decode(pred)}")
                total_count += 1

            del outputs
            torch.cuda.empty_cache()

    current_acc = round(correct_count / total_count, 2)
    print(f"Task accuracy: {current_acc}")
    return current_acc
