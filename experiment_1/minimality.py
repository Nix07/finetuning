import json
import random
import numpy as np
import math
import torch
import fire
from collections import defaultdict

from pp_utils import (
    get_model_and_tokenizer,
    load_eval_data,
    get_mean_activations,
    get_circuit,
    compute_pair_drop_values,
    get_head_significance_score,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

idx_to_group = {
    0: "struct_reader",
    1: "pos_transmitter",
    2: "pos_detector",
    3: "value_fetcher",
}
idx_to_pos = {0: -1, 1: 0, 2: 2, 3: 0}
minimal_circuit = defaultdict(list)


def set_seed(seed):
    """
    Sets the seed for random, numpy, and torch

    Args:
        seed (int): seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def minimality_main(
    datafile: str = "./data/dataset.jsonl",
    circuit_root_path: str = "../experiment_1/results/path_patching/llama_circuit",
    num_boxes: int = 7,
    model_name: str = "llama",
    num_samples: int = 100,
    batch_size: int = 100,
    n_value_fetcher: int = 101,  # Goat circuit: 101, FLoat circuit: 102, Llama circuit: 58
    n_pos_trans: int = 30,  # Goat circuit: 30, FLoat circuit: 30, Llama circuit: 10
    n_pos_detect: int = 50,  # Goat circuit: 50, FLoat circuit: 50, Llama circuit: 25
    n_struct_read: int = 40,  # Goat circuit: 40, FLoat circuit: 40, Llama circuit: 5
    percentage: float = 0.3,
    minimality_threshold: float = 0.01,
    seed: int = 10,  # Goat circuit: 56, FLoat circuit: 10, Llama circuit: 10
    results_path: str = "../experiment_1/results/minimality/llama_circuit",
):
    """
    Computes the minimality scores for the heads in the model

    Args:
        datafile (str): path to the datafile.
        circuit_root_path (str): path to the circuit components.
        num_boxes (int): number of boxes in the dataset.
        model_name (str): name of the model.
        num_samples (int): number of samples in the dataset.
        batch_size (int): batch size.
        n_value_fetcher (int): number of value fetcher heads.
        n_pos_trans (int): number of position transmitter heads.
        n_pos_detect (int): number of position detector heads.
        n_struct_read (int): number of structure reader heads.
        percentage (float): percentage of heads to consider.
        minimality_threshold (float): threshold for minimality.
        seed (int): seed value.
        results_path (str): path to store the results.
    """
    # Print the arguments
    print(f"DATAFILE: {datafile}")
    print(f"CIRCUIT_ROOT_PATH: {circuit_root_path}")
    print(f"NUM_BOXES: {num_boxes}")
    print(f"MODEL_NAME: {model_name}")
    print(f"NUM_SAMPLES: {num_samples}")
    print(f"BATCH_SIZE: {batch_size}")
    print(f"N_VALUE_FETCHER: {n_value_fetcher}")
    print(f"N_POS_TRANS: {n_pos_trans}")
    print(f"N_POS_DETECT: {n_pos_detect}")
    print(f"N_STRUCT_POS: {n_struct_read}")
    print(f"PERCENTAGE: {percentage}")
    print(f"SEED: {seed}")
    print(f"RESULTS_PATH: {results_path}")

    set_seed(seed)

    model, tokenizer = get_model_and_tokenizer(model_name)
    dataloader = load_eval_data(
        tokenizer=tokenizer,
        datafile=datafile,
        num_samples=num_samples,
        batch_size=batch_size,
    )

    mean_activations, modules = get_mean_activations(
        model=model,
        tokenizer=tokenizer,
        datafile=datafile,
        num_samples=num_samples,
        batch_size=batch_size,
    )

    (
        circuit_components,
        value_fetcher_heads,
        pos_trans_heads,
        pos_detect_heads,
        struct_read_heads,
    ) = get_circuit(
        model=model,
        circuit_root_path=circuit_root_path,
        n_value_fetcher=n_value_fetcher,
        n_pos_trans=n_pos_trans,
        n_pos_detect=n_pos_detect,
        n_struct_read=n_struct_read,
    )

    print(f"Value Fetcher Heads: {len(value_fetcher_heads)}")
    print(f"Position Transmitter Heads: {len(pos_trans_heads)}")
    print(f"Position Detector Heads: {len(pos_detect_heads)}")
    print(f"Structure Reader Heads: {len(struct_read_heads)}")

    print("Started Computing Minimality Scores...")
    for idx, head_group in enumerate(
        [struct_read_heads, pos_trans_heads, pos_detect_heads, value_fetcher_heads]
    ):
        print(f"{idx_to_group[idx]} Heads Started...")
        data = compute_pair_drop_values(
            model=model,
            heads=head_group,
            circuit_components=circuit_components,
            dataloader=dataloader,
            modules=modules,
            mean_activations=mean_activations,
            rel_pos=idx_to_pos[idx],
        )
        with open(
            f"{results_path}/{idx_to_group[idx]}.json", "w+", encoding="utf-8"
        ) as f:
            json.dump(data, f)

        ranked = defaultdict(list)
        for k_1 in data:
            for k_2 in data[k_1]:
                ranked[k_1].append((k_2, data[k_2][k_2] - data[k_1][k_2]))
        for k_1 in ranked:
            ranked[k_1].sort(key=(lambda x: x[1]), reverse=True)

        res = get_head_significance_score(
            model=model,
            heads=head_group,
            ranked=ranked,
            percentage=percentage,
            circuit_components=circuit_components,
            dataloader=dataloader,
            modules=modules,
            mean_activations=mean_activations,
            rel_pos=idx_to_pos[idx],
        )
        new_res = {}
        for k, v in res.items():
            new_res[str(k)] = v

        k = math.ceil(percentage * len(head_group))
        with open(
            f"{results_path}/{idx_to_group[idx]}_{k}_significance.json",
            "w+",
            encoding="utf-8",
        ) as f:
            json.dump(new_res, f)

        print(f"{idx_to_group[idx]} Heads Completed...")

        # Selecting heads with minimality score greater than threshold
        for k in new_res:
            if new_res[k][0] / new_res[k][1] - 1 >= minimality_threshold:
                if model.config.architectures[0] == "LlamaForCausalLM":
                    head = [int(k.split(".")[2]), int(k.split(",")[1][1:-1])]
                else:
                    head = [int(k.split(".")[4]), int(k.split(",")[1][1:-1])]
                minimal_circuit[idx_to_group[idx]].append(head)

    with open(
        f"{results_path}/{model_name}_circuit.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(minimal_circuit, f)

    print("Minimal Circuit Computed...")


if __name__ == "__main__":
    fire.Fire(minimality_main)
