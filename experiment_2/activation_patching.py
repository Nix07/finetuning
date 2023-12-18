import os
import random
import json
import sys
import fire
import torch
from collections import defaultdict
from functools import partial
from tqdm import tqdm

from baukit import TraceDict

from functionality_utils import (
    get_model_and_tokenizer,
    load_data_for_act_patching,
    activation_patching,
)

curr_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.abspath(os.path.join(curr_dir, os.pardir))
sys.path.append(parent_dir)
from data.data_utils import (
    positional_desiderata,
    object_value_desiderata,
    box_label_value_desiderata,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(20)
torch.manual_seed(20)

relative_pos = {
    "struct_reader": -1,
    "pos_transmitter": 0,
    "pos_detector": 2,
    "value_fetcher": 0,
}

desiderata_methods = {
    "positional": positional_desiderata,
    "object_value": object_value_desiderata,
    "box_label_value": box_label_value_desiderata,
}

results = defaultdict(str)


def act_patching_main(
    model_name: str = "llama",
    circuit_name: str = "llama",
    data_file: str = "./data/dataset.jsonl",
    object_file: str = "./data/objects.jsonl",
    num_samples: int = 500,
    batch_size: int = 32,
    output_dir: str = "./experiment_2/results/activation_patching",
):
    """
    Main function for activation patching experiments.

    Args:
        model_name (str): Name of the model to be used.
        data_file (str): Path to the dataset file.
        object_file (str): Path to the object file.
        num_samples (int): Number of samples to be generated.
        batch_size (int): Batch size for the dataloader.
        output_dir (str): Path to the output directory.
    """
    results[model_name] = defaultdict(dict)
    model, tokenizer = get_model_and_tokenizer(model_name)
    print("Model and Tokenizer loaded")

    for head_group, rel_pos in relative_pos.items():
        for desiderata in ["positional", "object_value", "box_label_value"]:
            with open(
                f"./experiment_2/results/DCM/{circuit_name}/{head_group}/{desiderata}/0.01.txt",
                "r",
                encoding="utf-8",
            ) as f:
                data = f.readlines()
                heads = json.loads(data[0].split(": ")[1])

            patching_heads = {rel_pos: heads}

            for loop_idx in tqdm(range(10)):
                raw_data = desiderata_methods[desiderata](
                    tokenizer=tokenizer,
                    num_samples=num_samples,
                    data_file=data_file,
                    object_file=object_file,
                    num_boxes=7,
                    alt_format=True,
                    correct_pred_indices=[],
                )

                dataloader = load_data_for_act_patching(
                    raw_data=raw_data, batch_size=batch_size
                )

                if model.config.architectures[0] == "LlamaForCausalLM":
                    modules = [
                        f"model.layers.{i}.self_attn.o_proj"
                        for i in range(model.config.num_hidden_layers)
                    ]
                else:
                    modules = [
                        f"base_model.model.model.layers.{i}.self_attn.o_proj"
                        for i in range(model.config.num_hidden_layers)
                    ]

                # Computing the activations of counterfactual examples
                source_cache = {}
                for bi, inputs in tqdm(
                    enumerate(dataloader), desc="counterfactual activations"
                ):
                    for k, v in inputs.items():
                        if v is not None and isinstance(v, torch.Tensor):
                            inputs[k] = v.to(model.device)

                    with TraceDict(model, modules, retain_input=True) as cache:
                        _ = model(inputs["source_input_ids"])

                    for module in modules:
                        if bi in source_cache:
                            source_cache[bi][module] = (
                                cache[module].input.detach().cpu()
                            )
                        else:
                            source_cache[bi] = {
                                module: cache[module].input.detach().cpu()
                            }
                print("Counterfactual activations computed\n")

                # Applying activation patching from counterfactual examples
                correct_count, total_count = 0, 0
                for bi, inputs in tqdm(
                    enumerate(dataloader), desc=f"patching loop[{loop_idx}]"
                ):
                    for k, v in inputs.items():
                        if v is not None and isinstance(v, torch.Tensor):
                            inputs[k] = v.to(model.device)

                    with TraceDict(
                        model,
                        modules,
                        retain_input=True,
                        edit_output=partial(
                            activation_patching,
                            model=model,
                            source_cache=source_cache,
                            patching_heads=patching_heads,
                            bi=bi,
                            input_tokens=inputs,
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
                print(
                    f"Head group: {head_group}, Desiderata: {desiderata}, Accuracy: {acc}"
                )

                if desiderata in results[model_name][head_group]:
                    results[model_name][head_group][desiderata].append(acc)
                else:
                    results[model_name][head_group][desiderata] = [acc]

                save_path = output_dir + f"/{model_name}_circuit_semantic_results.json"
                with open(save_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=4)


if __name__ == "__main__":
    fire.Fire(act_patching_main)
