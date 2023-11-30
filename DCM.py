import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, LlamaForCausalLM
from baukit import TraceDict
from einops import rearrange, einsum
from tqdm import tqdm
from functools import partial
from collections import defaultdict
import matplotlib.pyplot as plt
from datasets import Dataset
from torch.utils.data import DataLoader, SequentialSampler
import analysis_utils

# from model_aligner_script import load_data
from counterfactual_datasets.entity_tracking import *
from peft import PeftModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

# %%
print("Loading model...")
# path = "/home/local_nikhil/Projects/llama_weights/7B"
# tokenizer = AutoTokenizer.from_pretrained(path)
# model = AutoModelForCausalLM.from_pretrained(path).to(DEVICE)

base_model = "decapoda-research/llama-7b-hf"
lora_weights = "tiedong/goat-lora-7b"

tokenizer = LlamaTokenizer.from_pretrained(
    "hf-internal-testing/llama-tokenizer", padding_side="right"
)
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
tokenizer.pad_token_id = tokenizer.eos_token_id

circuit_path = "./minimality/new_circuit_heads.json"

model.eval()
for param in model.parameters():
    param.requires_grad_(False)

print("Model loaded.")
# %%
NUM_HEADS = model.config.num_attention_heads
HEAD_SIZE = model.config.hidden_size // NUM_HEADS


# %%
def load_data(raw_data, batch_size):
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


# %%
data_file_path = "./box_datasets/no_instructions/alternative/Random/7/train.jsonl"
object_file_path = "./box_datasets/filtered_objects_with_bnc_frequency.csv"

desiderata = {
    "positional": positional_desiderata,
    "object_value": object_value_desiderata,
    "box_label_value": box_label_value_desiderata,
}


def edit_output(
    inputs=None,
    output=None,
    layer=None,
    mask=None,
    from_activations=None,
    to_last_token_pos=None,
    from_last_token_pos=None,
    rel_pos=None,
    input_tokens=None,
):
    if "self_attn" in layer:
        inp = inputs[0]
        from_activations[layer] = from_activations[layer].to(DEVICE)

        # Computing the output of each head in this layer after the intervention
        for head_idx in range(NUM_HEADS):
            head_start = head_idx * HEAD_SIZE
            head_end = (head_idx + 1) * HEAD_SIZE

            if f"{layer}.{head_idx}" in mask_dict:
                abl_amt = mask[mask_dict[f"{layer}.{head_idx}"]]

                for batch in range(inp.shape[0]):
                    if rel_pos != -1:
                        intervention = (
                            abl_amt
                            * inp[
                                batch, to_last_token_pos[batch] - rel_pos, head_start:head_end
                            ].clone()
                            + (1 - abl_amt)
                            * from_activations[layer][
                                batch, from_last_token_pos[batch] - rel_pos, head_start:head_end
                            ]
                        )
                        inp[
                            batch, to_last_token_pos[batch] - rel_pos, head_start:head_end
                        ] = intervention
                    else:
                        base_prev_box_token_pos = analysis_utils.compute_prev_query_box_pos(
                            input_tokens["base_input_ids"][batch],
                            input_tokens["base_input_last_pos"][batch],
                        )
                        source_prev_box_token_pos = analysis_utils.compute_prev_query_box_pos(
                            input_tokens["source_input_ids"][batch],
                            input_tokens["source_input_last_pos"][batch],
                        )

                        intervention = (
                            abl_amt
                            * inp[batch, base_prev_box_token_pos, head_start:head_end].clone()
                            + (1 - abl_amt)
                            * from_activations[layer][
                                batch, source_prev_box_token_pos, head_start:head_end
                            ]
                        )
                        inp[batch, base_prev_box_token_pos, head_start:head_end] = intervention

        from_activations[layer] = from_activations[layer].to("cpu")

        weights = model.state_dict()[f"{layer}.weight"]
        mod_output = einsum(
            inp, weights, "batch seq_len hidden_size, d_model hidden_size -> batch seq_len d_model"
        )

        del weights
        torch.cuda.empty_cache()
        return mod_output

    else:
        assert False, "shouldn't be here"


relative_pos = {
    "direct_logit_heads": 0,
    "heads_affect_direct_logit": 0,
    "heads_at_query_box_pos": 2,
    "heads_at_prev_query_box_pos": -1,
}

for desideratum_name, desideratum_method in desiderata.items():
    raw_data = desideratum_method(
        tokenizer=tokenizer,
        num_samples=2000,
        data_file=data_file_path,
        object_file=object_file_path,
        num_boxes=7,
        alt_format=True,
        correct_pred_indices=[],
    )

    # %%
    objValueFetcher_train, objValueFetcher_eval, objValueFetcher_test = load_data(
        raw_data=raw_data, batch_size=32
    )
    desiderata_train = [objValueFetcher_train]
    desiderata_eval = [objValueFetcher_eval]
    desiderata_test = [objValueFetcher_test]

    circuit_components, head_groups = analysis_utils.get_circuit_components(model, circuit_path)

    if model.config.architectures[0] == "LlamaForCausalLM":
        modules = [f"model.layers.{layer}.self_attn.o_proj" for layer in range(32)]
    else:
        modules = [f"base_model.model.model.layers.{layer}.self_attn.o_proj" for layer in range(32)]

    from_activations_train = analysis_utils.load_activations(
        model, modules, desiderata_train, DEVICE
    )
    from_activations_eval = analysis_utils.load_activations(model, modules, desiderata_eval, DEVICE)
    from_activations_test = analysis_utils.load_activations(model, modules, desiderata_test, DEVICE)

    for head_group_name, head_group in tqdm(head_groups.items()):
        print(f"{desideratum_name}, {head_group_name} training started...")
        modules_w_heads = []
        for l, h in head_group:
            if model.config.architectures[0] == "LlamaForCausalLM":
                modules_w_heads.append(f"model.layers.{l}.self_attn.o_proj.{h}")
            else:
                modules_w_heads.append(f"base_model.model.model.layers.{l}.self_attn.o_proj.{h}")

        mask_dict = {module: i for i, module in enumerate(modules_w_heads)}

        mask = {}
        epochs = 2
        rel_pos = relative_pos[head_group_name]
        log_steps = 2
        eval_steps = 4
        save_path = f"./new_masks/goat-7b_new/{head_group_name}/{desideratum_name}/"

        # check if the path exists
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for lamb in [0.01]:
            mask[lamb] = torch.ones(
                len(modules_w_heads), requires_grad=True, device=DEVICE, dtype=torch.float
            )
            optimizer = torch.optim.Adam([mask[lamb]], lr=1e-1)
            eval_acc = -np.inf
            eval_heads = -np.inf

            for epoch in range(epochs):
                for di, desid_train in enumerate(desiderata_train):
                    for bi, inputs in enumerate(desid_train):
                        mask[lamb].data.clamp_(0, 1)
                        optimizer.zero_grad()

                        with TraceDict(
                            model,
                            modules,
                            edit_output=partial(
                                edit_output,
                                mask=mask[lamb],
                                from_activations=from_activations_train[di][bi],
                                to_last_token_pos=inputs["base_input_last_pos"],
                                from_last_token_pos=inputs["source_input_last_pos"],
                                rel_pos=rel_pos,
                                input_tokens=inputs,
                            ),
                        ) as _:
                            output = model(inputs["base_input_ids"].to(DEVICE))

                        target_logits = 0
                        for idx in range(inputs["base_input_ids"].size(0)):
                            target = inputs["labels"][idx]
                            target_logits += output.logits[
                                idx, inputs["base_input_last_pos"][idx], target
                            ]
                        target_logits /= inputs["base_input_ids"].size(0)

                        # maximize the target logits => minimize the negative target logits
                        # minimize the number of heads => maximize #ones in the mask
                        loss = -target_logits + lamb * torch.sum(1 - mask[lamb])

                        loss.backward()
                        optimizer.step()

                        if bi % log_steps == 0:
                            print(
                                f"epoch: {epoch}, bi: {bi}, Loss: {loss.item()}, Target logits: {target_logits.item()}"
                            )

                        if bi % eval_steps == 0:
                            with torch.inference_mode():
                                mask_data = mask[lamb].data.clone()
                                mask_data.clamp_(0, 1)
                                rounded = torch.round(mask_data)
                                heads = analysis_utils.compute_heads_from_mask(
                                    model, mask_dict, rounded
                                )
                                print(
                                    f"lamb: {lamb}, #Zero heads: {(rounded == 0).nonzero().shape[0]}"
                                )
                                print(heads)

                                correct, total = 0, 0
                                for eval_di, desid_eval in enumerate(desiderata_eval):
                                    for eval_bi, eval_inputs in enumerate(desid_eval):
                                        with TraceDict(
                                            model,
                                            modules,
                                            edit_output=partial(
                                                edit_output,
                                                mask=rounded,
                                                from_activations=from_activations_eval[eval_di][
                                                    eval_bi
                                                ],
                                                to_last_token_pos=eval_inputs[
                                                    "base_input_last_pos"
                                                ],
                                                from_last_token_pos=eval_inputs[
                                                    "source_input_last_pos"
                                                ],
                                                rel_pos=rel_pos,
                                                input_tokens=inputs,
                                            ),
                                        ) as _:
                                            eval_output = model(
                                                eval_inputs["base_input_ids"].to(DEVICE)
                                            )

                                        for idx in range(eval_inputs["base_input_ids"].size(0)):
                                            target = eval_inputs["labels"][idx]
                                            pred = torch.argmax(
                                                eval_output.logits[
                                                    idx, eval_inputs["base_input_last_pos"][idx]
                                                ]
                                            )

                                            if target == pred:
                                                correct += 1
                                            total += 1

                                        del eval_output
                                        torch.cuda.empty_cache()

                                acc = correct / total
                                if acc > eval_acc or (acc == eval_acc and len(heads) > eval_heads):
                                    print(acc, eval_acc, eval_heads, len(heads))
                                    eval_acc = acc
                                    eval_heads = len(heads)
                                    torch.save(mask[lamb].data, f"{save_path}/{lamb}")

                                print(f"lamb: {lamb}, Validation Accuracy: {acc}\n")

                    del output
                    torch.cuda.empty_cache()

        print(f"{desideratum_name}, {head_group_name} training finished...\n")

        print(f"{desideratum_name}, {head_group_name} testing started...")
        num_heads, valid_acc = [], []
        for lamb in [0.01]:
            with torch.no_grad():
                mask = torch.load(f"{save_path}/{lamb}")
                mask.clamp_(0, 1)
                rounded = torch.round(mask.data)
                heads = analysis_utils.compute_heads_from_mask(model, mask_dict, rounded)
                print(f"lamb: {lamb}, #Zero heads: {(rounded == 0).nonzero().shape[0]}")
                print(heads)

                correct, total = 0, 0
                for di, desid_test in enumerate(desiderata_test):
                    for bi, inputs in enumerate(desid_test):
                        with TraceDict(
                            model,
                            modules,
                            edit_output=partial(
                                edit_output,
                                mask=rounded,
                                from_activations=from_activations_test[di][bi],
                                to_last_token_pos=inputs["base_input_last_pos"],
                                from_last_token_pos=inputs["source_input_last_pos"],
                                rel_pos=rel_pos,
                                input_tokens=inputs,
                            ),
                        ) as _:
                            output = model(inputs["base_input_ids"].to(DEVICE))

                        for idx in range(inputs["base_input_ids"].size(0)):
                            target = inputs["labels"][idx]
                            pred = torch.argmax(
                                output.logits[idx, inputs["base_input_last_pos"][idx]]
                            )

                            if target == pred:
                                correct += 1
                            total += 1

                        del output
                        torch.cuda.empty_cache()

                num_heads.append((rounded == 0).nonzero().shape[0])
                valid_acc.append(correct / total)
                print(f"lamb: {lamb}, Test Accuracy: {correct / total}\n")

                with open(f"{save_path}/{lamb}.txt", "w") as f:
                    f.write(f"Heads: {heads}\nTest Accuracy: {correct / total}\n")

        print(f"{desideratum_name}, {head_group_name} testing finished...")
        print("---------------------------------------------")
