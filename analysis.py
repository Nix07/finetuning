# %%
import torch
from torch.nn import CosineSimilarity
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
from functools import partial
from baukit import TraceDict
from einops import rearrange, einsum
from plotly_utils import imshow
from tqdm import tqdm
import pysvelte
import importlib
import analysis_utils
from counterfactual_datasets.entity_tracking import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
importlib.reload(analysis_utils)
# %%
print("Model Loading...")
path = "./llama_7b/"
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path).to(device)
tokenizer.pad_token_id = tokenizer.eos_token_id

# %%
raw_data = modified_box_name_alignment_example_sampler(
    tokenizer,
    num_samples=6,
    data_file="./box_datasets/no_instructions/3/train.jsonl",
    architecture=model.config.architectures[0],
    object_file=None,
    num_ents_or_ops=3,
)

# %%
base_tokens = raw_data[0]
base_last_token_index = raw_data[1]
source_tokens = raw_data[2]
source_last_token_indices = raw_data[3]
correct_answer_token = raw_data[4]
# incorrect_answer_token = raw_data[6]

base_tokens = torch.tensor(base_tokens).to(device)
source_tokens = torch.tensor(source_tokens).to(device)

print("Data Generation Complete")


# %%
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
        input[bi, base_last_token_index[bi], head, :] = source_attn_output[layer][
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
                logits = torch.log_softmax(output.logits[bi, base_last_token_index[bi], :], dim=-1)
                logit_value += logits[correct_answer_token[bi][base_last_token_index[bi]]]

        logit_values[layer, head] = logit_value / batch_size

# %%
# Saving logit_values
torch.save(logit_values, "logit_values.pt")

# %%
# Load logit_values
# logit_values = torch.load("logit_values.pt")

# %%
# imshow((logit_values - torch.mean(logit_values)) / torch.std(logit_values))


# %%
# def compute_topk_components(patching_scores: torch.Tensor, k: int, largest=True):
#     """Computes the topk most influential components (i.e. heads) for patching."""
#     top_indices = torch.topk(patching_scores.flatten(), k, largest=largest).indices

#     # Convert the top_indices to 2D indices
#     row_indices = top_indices // patching_scores.shape[1]
#     col_indices = top_indices % patching_scores.shape[1]
#     top_components = torch.stack((row_indices, col_indices), dim=1)
#     # Get the top indices as a list of 2D indices (row, column)
#     top_components = top_components.tolist()
#     return top_components


# %%
# compute_topk_components(logit_values, 10, largest=True)
# %%
# layer = 21
# attn_scores = analysis_utils.get_attn_scores(model, source_tokens, layer)
# %%
# index = 4
# print(f"Layer: {layer}, Bi: {index}")
# pysvelte.AttentionMulti(
#     tokens=[tokenizer.decode(token) for token in source_tokens[index].cpu().tolist()],
#     attention=attn_scores[index].permute(1, 2, 0).cpu(),
# ).show()


# # %%
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
