# %%
import torch
from torch.nn import CosineSimilarity
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
from functools import partial
from baukit import TraceDict
from einops import rearrange, einsum
from collections import defaultdict
import matplotlib.pyplot as plt
from plotly_utils import imshow, scatter
from tqdm import tqdm

import pysvelte
import analysis_utils
from counterfactual_datasets.entity_tracking import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(10)

# %%
print("Model Loading...")
path = "./llama_7b/"
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path).to(device)
tokenizer.pad_token_id = tokenizer.eos_token_id

# %%
raw_data = box_index_aligner_examples(
    tokenizer,
    num_samples=60,
    data_file="./box_datasets/no_instructions/3/train.jsonl",
    # object_file="./box_datasets/objects_with_bnc_frequency.csv",
    architecture="LLaMAForCausalLM",
    num_ents_or_ops=3,
)

# %%
base_tokens = raw_data[0]
base_last_token_indices = raw_data[1]
source_tokens = raw_data[2]
source_last_token_indices = raw_data[3]
correct_answer_token = raw_data[4]
# incorrect_answer_token = raw_data[6]

base_tokens = torch.tensor(base_tokens).to(device)
source_tokens = torch.tensor(source_tokens).to(device)

print("Data Generation Complete")


# %%
###################################################
# Implementing Activation Patching
###################################################

hook_points = [
    f"model.layers.{layer}.self_attn.o_proj" for layer in range(model.config.num_hidden_layers)
]

with torch.no_grad():
    with TraceDict(model, hook_points, retain_input=True) as source_head_outputs:
        _ = model(source_tokens)

# _logits = _out.logits[0, base_last_token_index[0]]
# prob, pred = torch.max(_logits, dim=-1)
# pred = tokenizer.decode(pred)
# print(f"Pred: {pred}, Prob: {prob.item()}")

# %%
source_attn_output = {}
for layer in range(model.config.num_hidden_layers):
    source_attn_output[f"model.layers.{layer}.self_attn.o_proj"] = source_head_outputs[
        f"model.layers.{layer}.self_attn.o_proj"
    ].input

    source_attn_output[f"model.layers.{layer}.self_attn.o_proj"] = rearrange(
        source_attn_output[f"model.layers.{layer}.self_attn.o_proj"],
        "batch seq_len (num_heads d_head) -> batch seq_len num_heads d_head",
        num_heads=model.config.num_attention_heads,
    )


# %%
def patch_activation(inputs, output, layer, head):
    input = inputs[0]
    batch_size = input.size(0)

    input = rearrange(
        input,
        "batch seq_len (n_heads d_head) -> batch seq_len n_heads d_head",
        n_heads=model.config.num_attention_heads,
    )
    for bi in range(batch_size):
        input[bi, base_last_token_indices[bi], head, :] = source_attn_output[layer][
            bi, source_last_token_indices[bi], head, :
        ]
    input = rearrange(
        input,
        "batch seq_len n_heads d_head -> batch seq_len (n_heads d_head)",
        n_heads=model.config.num_attention_heads,
    )
    layer_index = int(layer.split(".")[2])
    w_o = model.model.layers[layer_index].self_attn.o_proj.weight
    output = einsum(
        input, w_o, "batch seq_len hidden_size, d_model hidden_size -> batch seq_len d_model"
    )

    return output


# %%
print("Patching Started...")
logit_values = torch.zeros(model.config.num_hidden_layers, model.config.num_attention_heads)
batch_size = base_tokens.size(0)

for layer in tqdm(range(model.config.num_hidden_layers)):
    hook_point = f"model.layers.{layer}.self_attn.o_proj"
    for head in range(model.config.num_attention_heads):
        with torch.no_grad():
            with TraceDict(
                model,
                [hook_point],
                retain_input=True,
                edit_output=partial(
                    patch_activation,
                    head=head,
                ),
            ) as _:
                output = model(base_tokens)

            logit_value = 0
            for bi in range(batch_size):
                logits = torch.log_softmax(
                    output.logits[bi, base_last_token_indices[bi], :], dim=-1
                )
                logit_value += logits[correct_answer_token[bi][base_last_token_indices[bi]]]

        logit_values[layer, head] = logit_value / batch_size

# %%
imshow(
    (logit_values - torch.mean(logit_values)) / torch.std(logit_values),
    # title="Query Box Reference Mover Heads",
    yaxis_title="Layer",
    xaxis_title="Head",
)

# %%
top_heads = analysis_utils.compute_topk_components(logit_values, 10, largest=True)

# %%
layer = 15
attn_scores = analysis_utils.get_attn_scores(model, source_tokens, layer)

# %%
index = 3
print(f"Layer: {layer}, Bi: {index}")
pysvelte.AttentionMulti(
    tokens=[tokenizer.decode(token) for token in source_tokens[index].cpu().tolist()],
    attention=attn_scores[index].permute(1, 2, 0).cpu(),
).show()


# %%
hook_point = [
    f"model.layers.{layer}.self_attn.o_proj" for layer in range(model.config.num_hidden_layers)
]
with torch.no_grad():
    with TraceDict(
        model,
        hook_point,
        retain_input=True,
    ) as cache:
        _ = model(source_tokens)

# %%
head_out_logit = torch.zeros(source_tokens.size(0))

query_box_pos = [raw_data[2][bi].index(29889) + 3 for bi in range(source_tokens.size(0))]
query_box = [raw_data[2][bi][query_box_pos[bi]] for bi in range(source_tokens.size(0))]

for layer, head in top_heads:
    hook_point = f"model.layers.{layer}.self_attn.o_proj"

    all_head_out = cache[hook_point].input
    batch_size = source_tokens.size(0)
    d_head = model.config.hidden_size // model.config.num_attention_heads

    decoder = torch.nn.Sequential(
        model.model.layers[layer].self_attn.o_proj,
        model.model.norm,
        model.lm_head,
        torch.nn.LogSoftmax(dim=-1),
    )

    start = head * d_head
    end = (head + 1) * d_head

    for bi in range(batch_size):
        head_output = all_head_out[bi, source_last_token_indices[bi], start:end]
        head_output = torch.concat(
            (
                torch.zeros(head * d_head).to(head_output.device),
                head_output,
                torch.zeros((model.config.num_attention_heads - head - 1) * d_head).to(
                    head_output.device
                ),
            ),
            dim=0,
        )
        head_unembed = decoder(head_output)
        head_out_logit[bi] = head_unembed[query_box[bi]]

    print(f"Layer: {layer}, head: {head}, Mean Box Logit: {head_out_logit.mean()}")


# %%
############################################
# Implementing path patching
############################################

# Run model on clean and corrupt prompts and cache attention heads activations
hook_points = [
    f"model.layers.{layer}.self_attn.o_proj" for layer in range(model.config.num_hidden_layers)
]
with torch.no_grad():
    # Step 1
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


# %%
def patching_heads(
    inputs,
    output,
    layer,
    sender_layer,
    sender_head,
    clean_last_token_indices,
    corrupt_last_token_indices,
):
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

        layer = int(layer.split(".")[2])
        if sender_layer == layer:
            for bi in range(batch_size):
                # Patch in the output of the sender head from corrupt run
                input[bi, clean_last_token_indices[bi], sender_head] = corrupt_head_outputs[
                    bi, corrupt_last_token_indices[bi], sender_head
                ]

            for bi in range(batch_size):
                # Patch in the output of all the heads, except sender, in this layer from clean run
                for head_ind in range(model.config.num_attention_heads):
                    if head_ind != sender_head:
                        input[bi, clean_last_token_indices[bi], head_ind] = clean_head_outputs[
                            bi, clean_last_token_indices[bi], head_ind
                        ]
        else:
            for bi in range(batch_size):
                # Patch in the output of all the heads in this layer from clean run
                input[bi, clean_last_token_indices[bi]] = clean_head_outputs[
                    bi, clean_last_token_indices[bi]
                ]

        input = rearrange(
            input,
            "batch seq_len n_heads d_head -> batch seq_len (n_heads d_head)",
            n_heads=model.config.num_attention_heads,
        )

        w_o = model.model.layers[layer].self_attn.o_proj.weight
        output = einsum(
            input, w_o, "batch seq_len hidden_size, d_model hidden_size -> batch seq_len d_model"
        )

    return output


# %%
def patching_receiver_heads(
    inputs, output, layer, patched_cache, receiver_heads, clean_last_token_indices
):
    input = inputs[0]
    batch_size = input.size(0)
    receiver_heads_in_curr_layer = [h for l, h in receiver_heads if l == int(layer.split(".")[2])]

    input = rearrange(
        input,
        "batch seq_len (n_heads d_head) -> batch seq_len n_heads d_head",
        n_heads=model.config.num_attention_heads,
    )
    patched_head_outputs = rearrange(
        patched_cache[layer].input,
        "batch seq_len (n_heads d_head) -> batch seq_len n_heads d_head",
        n_heads=model.config.num_attention_heads,
    )

    # Patch in the input of the receiver heads from patched run
    for receiver_head in receiver_heads_in_curr_layer:
        for bi in range(batch_size):
            input[bi, clean_last_token_indices[bi], receiver_head] = patched_head_outputs[
                bi, clean_last_token_indices[bi], receiver_head
            ]

    input = rearrange(
        input,
        "batch seq_len n_heads d_head -> batch seq_len (n_heads d_head)",
        n_heads=model.config.num_attention_heads,
    )
    w = model.state_dict()[f"{layer}.weight"]
    output = einsum(
        input, w, "batch seq_len hidden_size, d_model hidden_size -> batch seq_len d_model"
    )

    return output


# %%
receiver_layers = [
    "model.layers.23.self_attn.q_proj",
    "model.layers.21.self_attn.q_proj",
    "model.layers.22.self_attn.q_proj",
    "model.layers.24.self_attn.q_proj",
]

logit_values = torch.zeros(model.config.num_hidden_layers, model.config.num_attention_heads)
batch_size = base_tokens.size(0)
for layer in tqdm(range(model.config.num_hidden_layers)):
    for head in range(model.config.num_attention_heads):
        with torch.no_grad():
            # Step 2
            with TraceDict(
                model,
                hook_points + receiver_layers,
                retain_input=True,
                edit_output=partial(
                    patching_heads,
                    sender_layer=layer,
                    sender_head=head,
                    clean_last_token_indices=base_last_token_indices,
                    corrupt_last_token_indices=source_last_token_indices,
                ),
            ) as patched_cache:
                _ = model(base_tokens)

            # Step 3
            with TraceDict(
                model,
                receiver_layers,
                retain_input=True,
                edit_output=partial(
                    patching_receiver_heads,
                    patched_cache=patched_cache,
                    receiver_heads=object_fetcher_heads,
                    clean_last_token_indices=base_last_token_indices,
                ),
            ) as _:
                patched_out = model(base_tokens)
            
            logits = torch.log_softmax(
                patched_out.logits[bi, base_last_token_indices[bi], :], dim=-1
            )
            logit_value += logits[correct_answer_token[bi][base_last_token_indices[bi]]]

        logit_values[layer, head] = logit_value / batch_size

# %%
logit_values = torch.load("object_fetcher_heads_act_patching.pt")
# %%
imshow(
    (logit_values - torch.mean(logit_values)) / torch.std(logit_values),
    # title="Query Box Reference Mover Heads",
    # yaxis_title="Layer",
    # xaxis_title="Head",
)

# %%
object_fetcher_heads = analysis_utils.compute_topk_components(logit_values, 10, largest=True)
print(object_fetcher_heads)

# %%
layer = 21
attn_scores = analysis_utils.get_attn_scores(model, base_tokens, layer)
# %%
index = 3
print(f"Layer: {layer}, Bi: {index}")
pysvelte.AttentionMulti(
    tokens=[tokenizer.decode(token) for token in base_tokens[index].cpu().tolist()],
    attention=attn_scores[index].permute(1, 2, 0).cpu(),
).show()
# %%
# Computing average attention scores to correct object by top 5 object fetcher heads

scores = defaultdict(list)
for layer in range(model.config.num_hidden_layers):
    attn_scores = analysis_utils.get_attn_scores(model, base_tokens, layer)

    for head in range(model.config.num_attention_heads):
        for bi in range(base_tokens.size(0)):
            correct_object = correct_answer_token[bi][base_last_token_indices[bi]]
            correct_object_pos = base_tokens[bi].tolist().index(correct_object)

            scores[(layer, head)].append(
                attn_scores[bi, head, base_last_token_indices[bi], correct_object_pos].item()
            )

        # scores[(layer, head)] /= source_tokens.size(0)

# %%
hook_point = [
    f"model.layers.{layer}.self_attn.o_proj" for layer in range(model.config.num_hidden_layers)
]
with torch.no_grad():
    with TraceDict(
        model,
        hook_point,
        retain_input=True,
    ) as cache:
        _ = model(base_tokens)

# %%
# Computing the logit value of correct object written by object fetcher heads
direct_logit_values = defaultdict(list)
for layer in tqdm(range(model.config.num_hidden_layers)):
    for head in range(model.config.num_attention_heads):
        hook_point = f"model.layers.{layer}.self_attn.o_proj"

        all_head_out = cache[hook_point].input
        batch_size = base_tokens.size(0)
        d_head = model.config.hidden_size // model.config.num_attention_heads

        decoder = torch.nn.Sequential(
            model.model.layers[layer].self_attn.o_proj,
            model.model.norm,
            model.lm_head,
        )

        start = head * d_head
        end = (head + 1) * d_head

        head_out_logit = torch.zeros(batch_size)
        for bi in range(batch_size):
            head_output = all_head_out[bi, base_last_token_indices[bi], start:end]
            head_output = torch.concat(
                (
                    torch.zeros(head * d_head).to(head_output.device),
                    head_output,
                    torch.zeros((model.config.num_attention_heads - head - 1) * d_head).to(
                        head_output.device
                    ),
                ),
                dim=0,
            )
            head_unembed = decoder(head_output)
            correct_object = correct_answer_token[bi][base_last_token_indices[bi]]
            head_out_logit[bi] = head_unembed[correct_object]
            direct_logit_values[(layer, head)].append(head_out_logit[bi].item())

        # direct_logit_values[(layer, head)] = head_out_logit.mean().item()

# %%
# Plot a scatter plot of one top object fetcher head with direct logit values vs attention scores
# layer, head = (10, 10)
# np.random.seed(50)
for layer, head in object_fetcher_heads:
    x = scores[(layer, head)]
    y = direct_logit_values[(layer, head)]
    plt.scatter(x, y, alpha=0.5, s=40, label=f"({layer}, {head})")

    plt.xlabel("Attention Score")
    plt.ylabel("Correct Object Logit Value (log softmax)")
    plt.title("Direct Logit Value vs Attention Score")
    plt.grid(True)
    plt.xlim(0, 1)
    plt.legend()
    plt.show()

# Generate 10 random heads
# layers = np.random.choice(model.config.num_hidden_layers, 10)
# heads = np.random.choice(model.config.num_attention_heads, 10)
# for layer, head in zip(layers, heads):
#     x = scores[(layer, head)]
#     y = direct_logit_values[(layer, head)]
#     plt.scatter(x, y, alpha=0.7, c="pink", s=20, marker="x", label="Random 10 Heads")



# %%
###############################################################
# Direct logit attribution to identify object fetcher heads
###############################################################

hook_point = [
    f"model.layers.{layer}.self_attn.o_proj" for layer in range(model.config.num_hidden_layers)
]
with torch.no_grad():
    with TraceDict(
        model,
        hook_point,
        retain_input=True,
    ) as cache:
        _ = model(base_tokens)

batch_size = base_tokens.size(0)
d_head = model.config.hidden_size // model.config.num_attention_heads

direct_logit_values = torch.zeros(
    (model.config.num_hidden_layers, model.config.num_attention_heads)
).to(model.device)
for layer in tqdm(range(model.config.num_hidden_layers)):
    for head in range(model.config.num_attention_heads):
        hook_point = f"model.layers.{layer}.self_attn.o_proj"
        all_head_out = cache[hook_point].input
        decoder = torch.nn.Sequential(
            model.model.layers[layer].self_attn.o_proj,
            model.model.norm,
            model.lm_head,
            torch.nn.LogSoftmax(dim=-1),
        )

        start = head * d_head
        end = (head + 1) * d_head

        for bi in range(batch_size):
            head_output = all_head_out[bi, base_last_token_indices[bi], start:end]
            head_output = torch.concat(
                (
                    torch.zeros(head * d_head).to(head_output.device),
                    head_output,
                    torch.zeros((model.config.num_attention_heads - head - 1) * d_head).to(
                        head_output.device
                    ),
                ),
                dim=0,
            )
            head_unembed = decoder(head_output)
            correct_object = correct_answer_token[bi][base_last_token_indices[bi]]
            direct_logit_values[(layer, head)] += torch.dot(head_unembed, model.lm_head.weight[correct_object])

        direct_logit_values[(layer, head)] = direct_logit_values[(layer, head)] / batch_size

# %%
imshow(
    (direct_logit_values - torch.mean(logit_values)) / torch.std(logit_values),
    # title="Query Box Reference Mover Heads",
    # yaxis_title="Layer",
    # xaxis_title="Head",
)


# %%
# pc_sim = []
# pc_index = 0
# with torch.no_grad():
#     for layer in range(32):
#         checkpoint_fixed = torch.load(
#             f"results/llama_7b_modified_query_box_identity/seed.42.rel.0.layer.{layer}/pytorch-rotate-last.bin"
#         )
#         rotate_matrix_fixed = checkpoint_fixed["rotate_layer"]["parametrizations.weight.original"]

#         boundary_end = int(checkpoint_fixed["intervention_boundaries"].item() * 4096)
#         U_fixed, S_fixed, V_fixed = torch.linalg.svd(rotate_matrix_fixed[:, :boundary_end])

#         checkpoint_flexible = torch.load(
#             f"results/llama_7b_modified_query_box_flexible_intervention_boundary/seed.42.rel.0.layer.{layer}/pytorch-rotate-last.bin"
#         )
#         rotate_matrix_flexible = checkpoint_flexible["rotate_layer"][
#             "parametrizations.weight.original"
#         ]

#         boundary_start = int(checkpoint_flexible["intervention_boundaries"][0].item() * 4096)
#         boundary_end = int(checkpoint_flexible["intervention_boundaries"][1].item() * 4096)
#         U_flexible, S_flexible, V_flexible = torch.linalg.svd(
#             rotate_matrix_flexible[:, boundary_start:boundary_end]
#         )

#         cos = CosineSimilarity(dim=0)
#         pc_sim.append(abs(cos(U_fixed[:, pc_index], U_flexible[:, pc_index])).cpu().item())
#         # print(f"Layer: {layer}, Sim: {cos(U_fixed[:, pc_index], U_flexible[:, pc_index])}")

# # %%
# # Compute cosine similartiy of two random vectors of size 4096
# random_sim = []
# for _ in range(100):
#     random_sim.append(abs(cos(torch.randn(4096), torch.randn(4096))).cpu().item())
# random_sim = sum(random_sim) / len(random_sim)
# print(f"Random vector similarity: {random_sim}")

# # %%
# fig, ax1 = plt.subplots()
# ax1.plot(pc_sim, "o:", label=f"PC{pc_index+1}")
# ax1.set_xlabel("Layers")
# ax1.set_ylabel("Cosine Similarity")
# ax1.set_title(f"Cosine Similarity of PC{pc_index+1} (fixed and flexible boundaries) vs. Layers")
# # Horizontal line at random_sim
# ax1.axhline(y=random_sim, color="r", linestyle="-", label="Random vectors similarity")
# ax1.set_ylim([0, 1])
# ax1.legend()
# plt.show()


# # %%
# with torch.no_grad():
#     boundaries = []
#     for layer in range(32):
#         with open(
#             f"results/llama_7b_modified_query_box_identity/seed.42.rel.0.layer.{layer}/train_log.txt"
#         ) as file:
#             data = file.readlines()
#             boundary = float(data[-1].split(",")[-1][:-1]) * 4096
#             boundaries.append(boundary)

# # Plot the boundaries
# fig, ax1 = plt.subplots()
# ax1.plot(boundaries, "o:", label=f"PC{pc_index+1}")
# ax1.set_xlabel("Layers")
# ax1.set_ylabel("Intervention Boundary (dimensions)")
# ax1.set_title(f"Intervention Boundary [0, end] vs. Layers")
# ax1.set_ylim([0, 4200])
# ax1.legend()
# plt.show()
# # %%

# with torch.no_grad():
#     boundary_start, boundary_end = [], []
#     for layer in range(32):
#         with open(
#             f"results/llama_7b_modified_query_box_flexible_intervention_boundary/seed.42.rel.0.layer.{layer}/train_log.txt"
#         ) as file:
#             data = file.readlines()
#             boundary_start.append(float(data[-1].split(",")[-2]) * 4096)
#             boundary_end.append(float(data[-1].split(",")[-1][:-1]) * 4096)

# # Plot the boundaries
# fig, ax1 = plt.subplots()
# ax1.plot(boundary_start, label="boundary_start")
# ax1.plot(boundary_end, label="boundary_end")
# ax1.set_xlabel("Layers")
# ax1.set_ylabel("Intervention Boundary (dimensions)")
# ax1.set_title(f"Intervention Boundary [0, end] vs. Layers")
# ax1.set_ylim([0, 4200])
# ax1.legend()
# plt.show()
# # %%

# %%
