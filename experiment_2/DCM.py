import os
import sys
import fire
import torch

import numpy as np
from baukit import TraceDict
from einops import einsum
from tqdm import tqdm
from functools import partial

from functionality_utils import (
    get_model_and_tokenizer,
    get_data,
    edit_output,
    get_circuit_components,
    load_activations,
    compute_heads_from_mask,
)

curr_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.abspath(os.path.join(curr_dir, os.pardir))
sys.path.append(parent_dir)
from data.data_utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(20)

relative_pos = {
    "struct_reader": -1,
    "pos_transmitter": 0,
    "pos_detector": 2,
    "value_fetcher": 0,
}

org_desiderata = {
    "positional": positional_desiderata,
    "object_value": object_value_desiderata,
    "box_label_value": box_label_value_desiderata,
}

additional_desiderata = {
    "random_text_start": add_raw_text_at_start,
    "random_text_end": add_raw_text_at_end,
    "add_tokens_btw_box_and_obj": additional_token_btw_box_and_object,
    "add_seg_start": add_segment_at_start,
    "add_seg_end": add_segment_at_end,
    "add_box_before_correct_segment": add_box_before_correct_segment,
    "incorrect_segment": diff_index_query_box,
    "altered_box_obj_order": box_object_altered_order,
    "altered_box_obj_association": alter_box_object_association,
    "no_comma": remove_comma_desiderata,
    "add_comma": add_comma_after_object,
}


def dcm_main(
    model_name: str = "llama",
    circuit_path: str = "../experiment_1/results/circuits/llama_circuit.json",
    batch_size: int = 32,
    data_file: str = "../data/dataset.jsonl",
    object_file: str = "../data/objects.csv",
    epochs: int = 2,
    log_steps: int = 2,
    eval_steps: int = 4,
    output_dir: str = "../experiment_2/results/DCM/",
    lambs: list = [0.01],
    use_add_desiderata: bool = False,
):
    """
    Main function for the DCM experiment.

    Args:
        model (str): Name of the model to load.
    """

    model, tokenizer = get_model_and_tokenizer(model_name)

    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    print("Model and Tokenizer loaded...\n")

    if use_add_desiderata:
        desiderata = additional_desiderata.copy()
    else:
        desiderata = org_desiderata.copy()

    for desid_name, desid_method in desiderata.items():
        print("Starting with desideratum: ", desid_name)
        desideratum_train, desideratum_eval, desideratum_test = get_data(
            desid_method, tokenizer, data_file, object_file, batch_size
        )

        _, head_groups = get_circuit_components(model, circuit_path)

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

        print("Computing counterfactual example activations...")
        from_activations_train = load_activations(
            model, modules, desideratum_train, device
        )
        from_activations_eval = load_activations(
            model, modules, desideratum_eval, device
        )
        from_activations_test = load_activations(
            model, modules, desideratum_test, device
        )
        print("Counterfactual example activations computed...\n")

        for head_group_name, head_group in tqdm(head_groups.items()):
            if use_add_desiderata:
                if head_group_name != "pos_transmitter":
                    continue
            print(f"{desid_name}, {head_group_name} training started...")
            modules_w_heads = []
            for l, h in head_group:
                if model.config.architectures[0] == "LlamaForCausalLM":
                    modules_w_heads.append(f"model.layers.{l}.self_attn.o_proj.{h}")
                else:
                    modules_w_heads.append(
                        f"base_model.model.model.layers.{l}.self_attn.o_proj.{h}"
                    )

            mask_dict = {module: i for i, module in enumerate(modules_w_heads)}

            mask = {}
            rel_pos = relative_pos[head_group_name]
            save_path = (
                output_dir + f"{model_name}_circuit/{head_group_name}/{desid_name}/"
            )
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            for lamb in lambs:
                mask[lamb] = torch.ones(
                    len(modules_w_heads),
                    requires_grad=True,
                    device=device,
                    dtype=torch.float,
                )
                optimizer = torch.optim.Adam([mask[lamb]], lr=1e-1)
                eval_acc = -np.inf
                eval_heads = -np.inf

                for epoch in range(epochs):
                    for di, desid_train in enumerate(desideratum_train):
                        for bi, inputs in enumerate(desid_train):
                            mask[lamb].data.clamp_(0, 1)
                            optimizer.zero_grad()

                            with TraceDict(
                                model,
                                modules,
                                edit_output=partial(
                                    edit_output,
                                    model=model,
                                    mask=mask[lamb],
                                    from_activations=from_activations_train[di][bi],
                                    to_last_token_pos=inputs["base_input_last_pos"],
                                    from_last_token_pos=inputs["source_input_last_pos"],
                                    rel_pos=rel_pos,
                                    input_tokens=inputs,
                                    device=device,
                                    mask_dict=mask_dict,
                                ),
                            ) as _:
                                output = model(inputs["base_input_ids"].to(device))

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
                                    heads = compute_heads_from_mask(
                                        model, mask_dict, rounded
                                    )
                                    print(
                                        f"lamb: {lamb}, #Zero heads: {(rounded == 0).nonzero().shape[0]}"
                                    )
                                    print(heads)

                                    correct, total = 0, 0
                                    for eval_di, desid_eval in enumerate(
                                        desideratum_eval
                                    ):
                                        for eval_bi, eval_inputs in enumerate(
                                            desid_eval
                                        ):
                                            with TraceDict(
                                                model,
                                                modules,
                                                edit_output=partial(
                                                    edit_output,
                                                    model=model,
                                                    mask=rounded,
                                                    from_activations=from_activations_eval[
                                                        eval_di
                                                    ][
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
                                                    device=device,
                                                    mask_dict=mask_dict,
                                                ),
                                            ) as _:
                                                eval_output = model(
                                                    eval_inputs["base_input_ids"].to(
                                                        device
                                                    )
                                                )

                                            for idx in range(
                                                eval_inputs["base_input_ids"].size(0)
                                            ):
                                                target = eval_inputs["labels"][idx]
                                                pred = torch.argmax(
                                                    eval_output.logits[
                                                        idx,
                                                        eval_inputs[
                                                            "base_input_last_pos"
                                                        ][idx],
                                                    ]
                                                )

                                                if target == pred:
                                                    correct += 1
                                                total += 1

                                            del eval_output
                                            torch.cuda.empty_cache()

                                    acc = correct / total
                                    if acc > eval_acc or (
                                        acc == eval_acc and len(heads) > eval_heads
                                    ):
                                        print(acc, eval_acc, eval_heads, len(heads))
                                        eval_acc = acc
                                        eval_heads = len(heads)
                                        torch.save(
                                            mask[lamb].data, f"{save_path}/{lamb}"
                                        )

                                    print(f"lamb: {lamb}, Validation Accuracy: {acc}\n")

                        del output
                        torch.cuda.empty_cache()

            print(f"{desid_name}, {head_group_name} training finished...\n")

            print(f"{desid_name}, {head_group_name} testing started...")
            num_heads, valid_acc = [], []
            for lamb in [0.01]:
                with torch.no_grad():
                    mask = torch.load(f"{save_path}/{lamb}")
                    mask.clamp_(0, 1)
                    rounded = torch.round(mask.data)
                    heads = compute_heads_from_mask(model, mask_dict, rounded)
                    print(
                        f"lamb: {lamb}, #Zero heads: {(rounded == 0).nonzero().shape[0]}"
                    )
                    print(heads)

                    correct, total = 0, 0
                    for di, desid_test in enumerate(desideratum_test):
                        for bi, inputs in enumerate(desid_test):
                            with TraceDict(
                                model,
                                modules,
                                edit_output=partial(
                                    edit_output,
                                    model=model,
                                    mask=rounded,
                                    from_activations=from_activations_test[di][bi],
                                    to_last_token_pos=inputs["base_input_last_pos"],
                                    from_last_token_pos=inputs["source_input_last_pos"],
                                    rel_pos=rel_pos,
                                    input_tokens=inputs,
                                    device=device,
                                    mask_dict=mask_dict,
                                ),
                            ) as _:
                                output = model(inputs["base_input_ids"].to(device))

                            for idx in range(inputs["base_input_ids"].size(0)):
                                target = inputs["labels"][idx]
                                pred = torch.argmax(
                                    output.logits[
                                        idx, inputs["base_input_last_pos"][idx]
                                    ]
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

            print(f"{desid_name}, {head_group_name} testing finished...")
            print("---------------------------------------------")


if __name__ == "__main__":
    fire.Fire(dcm_main)
