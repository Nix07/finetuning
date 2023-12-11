import json
import random
import numpy as np
import math
import torch
import fire
from collections import defaultdict

from pp_utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

idx_to_group = {
    0: "struct_reader",
    1: "pos_transmitter",
    2: "pos_detector",
    3: "value_fetcher",
}
idx_to_filename = {
    0: "llama_heads_at_prev_box_pos",
    1: "heads_affecting_direct_logit_heads",
    2: "head_at_query_box_token_new",
    3: "direct_logit_heads",
}
idx_to_pos = {0: -1, 1: 0, 2: 2, 3: 0}


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
    datafile: str = "../box_datasets/no_instructions/alternative/Random/7/train.jsonl",
    circuit_root_path: str = None,
    num_boxes: int = 7,
    model_name: str = "llama",
    num_samples: int = 100,
    batch_size: int = 100,
    n_value_fetcher: int = 66,
    n_pos_trans: int = 15,
    n_pos_detect: int = 30,
    n_struct_read: int = 5,
    percentage: float = 0.3,
    seed: int = 10,
    results_path: str = "./results/",
):
    """
    Computes the minimality scores for the heads in the model

    Args:
        datafile (str, optional): path to the datafile. Defaults to "../box_datasets/no_instructions/alternative/Random/7/train.jsonl".
        circuit_root_path (str, optional): path to the circuit components. Defaults to None.
        num_boxes (int, optional): number of boxes in the dataset. Defaults to 7.
        model_name (str, optional): name of the model. Defaults to "llama".
        num_samples (int, optional): number of samples in the dataset. Defaults to 100.
        batch_size (int, optional): batch size. Defaults to 100.
        n_value_fetcher (int, optional): number of value fetcher heads. Defaults to 66.
        n_pos_trans (int, optional): number of position transmitter heads. Defaults to 15.
        n_pos_detect (int, optional): number of position detector heads. Defaults to 30.
        n_struct_read (int, optional): number of structure reader heads. Defaults to 5.
        percentage (float, optional): percentage of heads to consider. Defaults to 0.3.
        seed (int, optional): seed value. Defaults to 10.
        results_path (str, optional): path to store the results. Defaults to "./results/".
    """
    # Print the arguments
    print("Arguments:")
    print(f"datafile: {datafile}")
    print(f"circuit_root_path: {circuit_root_path}")
    print(f"num_boxes: {num_boxes}")
    print(f"model_name: {model_name}")
    print(f"num_samples: {num_samples}")
    print(f"batch_size: {batch_size}")
    print(f"n_value_fetcher: {n_value_fetcher}")
    print(f"n_pos_trans: {n_pos_trans}")
    print(f"n_pos_detect: {n_pos_detect}")
    print(f"n_struct_read: {n_struct_read}")
    print(f"percentage: {percentage}")
    print(f"seed: {seed}")
    print(f"results_path: {results_path}")

    set_seed(seed)

    model, tokenizer = get_model_and_tokenizer(model_name)
    dataloader = loal_eval_data(
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

    print("Started Computing Functionality Sets...")
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

    print(
        len(value_fetcher_heads),
        len(pos_trans_heads),
        len(pos_detect_heads),
        len(struct_read_heads),
    )

    for idx, head_group in enumerate([struct_read_heads, pos_trans_heads, pos_detect_heads, value_fetcher_heads]):
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
        with open(f"{results_path}/{idx_to_filename[idx]}.json", "w") as f:
            json.dump(data, f)
        # print("Completed Computing Pair Drop Values...")

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
        for k in res:
            new_res[str(k)] = res[k]

        k = math.ceil(percentage * len(head_group))
        with open(
            f"{results_path}/{idx_to_filename[idx]}_{k}_significance.json", "w"
        ) as f:
            json.dump(new_res, f)

        print(f"{idx_to_group[idx]} Heads Completed...")

    print("Completed Computing Functionality Sets...")


if __name__ == "__main__":
    fire.Fire(minimality_main)
