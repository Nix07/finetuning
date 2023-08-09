# %%
from transformers import (
    set_seed,
    AutoConfig,
    get_linear_schedule_with_warmup,
    AutoTokenizer,
    AutoModelForCausalLM,
)
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from datasets import Dataset
import os, random, argparse, sys, torch
from models.configuration_alignable_model import AlignableLlamaConfig
from counterfactual_datasets.entity_tracking import *
from trainer import Aligner, CACHE_DIR
from torch.utils.data import DataLoader, SequentialSampler
from models.modelings_alignable import AutoAlignableModel
from tqdm import tqdm

from transformers.utils import logging

logging.set_verbosity_info()
logger = logging.get_logger("transformers")

import warnings

warnings.filterwarnings("ignore")

set_seed(42)

# %%
# TODO: Checking model's performance on the given task

# %%
# TODO: Saving the model and tokenizer with additional config


# %%
def load_data(
    tokenizer,
    data_size,
    aligner_func,
    data_file,
    architecture,
    num_ents_or_ops,
    batch_size,
    object_file,
):
    raw_data = alignment_example_sampler(
        tokenizer,
        data_size,
        aligner_func,
        data_file,
        num_ents_or_ops,
        architecture=architecture,
        object_file=object_file,
    )

    train_size = int(0.8 * len(raw_data[0]))
    eval_size = int(0.1 * len(raw_data[0]))

    print("Train size: ", train_size)
    print("Eval size: ", eval_size)
    print("Test size: ", len(raw_data[0]) - train_size - eval_size)

    raw_train = (
        raw_data[0][:train_size],
        raw_data[1][:train_size],
        raw_data[2][:train_size],
        raw_data[3][:train_size],
        raw_data[4][:train_size],
        raw_data[5][:train_size],
        raw_data[6][:train_size],
    )
    raw_eval = (
        raw_data[0][train_size : train_size + eval_size],
        raw_data[1][train_size : train_size + eval_size],
        raw_data[2][train_size : train_size + eval_size],
        raw_data[3][train_size : train_size + eval_size],
        raw_data[4][train_size : train_size + eval_size],
        raw_data[5][train_size : train_size + eval_size],
        raw_data[6][train_size : train_size + eval_size],
    )
    raw_test = (
        raw_data[0][train_size + eval_size :],
        raw_data[1][train_size + eval_size :],
        raw_data[2][train_size + eval_size :],
        raw_data[3][train_size + eval_size :],
        raw_data[4][train_size + eval_size :],
        raw_data[5][train_size + eval_size :],
        raw_data[6][train_size + eval_size :],
    )

    train_dataset = Dataset.from_dict(
        {
            "base_input_ids": raw_train[0],
            "base_input_last_pos": raw_train[1],
            "source_input_ids": raw_train[2],
            "source_input_last_pos": raw_train[3],
            "labels": raw_train[4],
            "incorrect_objects": raw_train[5],
            "intervention_ids": raw_train[6],
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
            "incorrect_objects": raw_eval[5],
            "intervention_ids": raw_eval[6],
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
            "incorrect_objects": raw_test[5],
            "intervention_ids": raw_test[6],
        }
    ).with_format("torch")
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
    )

    return train_dataloader, eval_dataloader, test_dataloader


# %%
def align_model(
    aligning_layers,
    layers_interval,
    model_name_or_path,
    train_dataloader,
    eval_dataloader,
    test_dataloader,
    num_train_epochs,
    batch_size,
    train_log_steps,
    seed,
    output_dir,
):
    for rel_pos in range(0, 4):
        for layer in range(0, aligning_layers, layers_interval):
            print(f"Starting traing for layer: {layer}")
            alignment_config = {
                "layer": layer,
                "rel_pos": rel_pos,
            }
            model = AutoAlignableModel.from_pretrained(
                model_name_or_path,
                alignment_config=alignment_config,
                torch_dtype=torch.float32,
                cache_dir=CACHE_DIR,
            )
            _ = model.to("cuda")  # first GPU

            # Training configuration

            # set off the gradients among all other layers.
            for name, param in model.named_parameters():
                if "rotate_layer" not in name and "intervention_boundaries" not in name:
                    param.requires_grad = False
                else:
                    logger.info(f"Requiring gradients on layer: {name}")
            t_total = int(len(train_dataloader) * num_train_epochs)
            warm_up_steps = 0.1 * t_total
            optimizer = torch.optim.Adam(
                [
                    {"params": model.model.rotate_layer.parameters()},
                    {"params": model.model.intervention_boundaries, "lr": 1e-2},
                ],
                lr=1e-3,
            )
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=warm_up_steps, num_training_steps=t_total
            )

            def compute_metrics(eval_preds, eval_labels, incorrect_objects):
                # eval_preds: (#batch, vocab_size)
                # eval_labels: (#batch)
                total_count = 0
                correct_count = 0
                for pred, correct_object, incorrect_objs in zip(
                    eval_preds, eval_labels, incorrect_objects
                ):
                    first_incorrect_object = incorrect_objs[0]
                    second_incorrect_object = incorrect_objs[1]

                    if (
                        pred[correct_object] >= pred[first_incorrect_object]
                        and pred[correct_object] >= pred[second_incorrect_object]
                    ):
                        correct_count += 1

                    total_count += 1

                accuracy = round(correct_count / total_count, 2)
                return {"accuracy": accuracy}

            run_name = f"seed.{seed}.rel.{rel_pos}.layer.{layer}/"
            if not os.path.exists(f"{output_dir}"):
                os.mkdir(f"{output_dir}")
            os.environ["WANDB_PROJECT"] = f"Boundless-DAS"
            output_file_path = os.path.join(f"{output_dir}", run_name)
            if not os.path.exists(output_file_path):
                os.mkdir(output_file_path)

            aligner = Aligner(
                model,
                logger=logger,
                is_wandb=False,
                is_master=True,
                n_gpu=1,
                model_name=run_name,
                device="cuda",
                compute_metrics=compute_metrics,
            )

            aligner.train(
                train_dataloader,
                eval_dataloader,
                test_dataloader,
                optimizer,
                scheduler,
                log_step=train_log_steps,
                valid_steps=20,
                output_dir=output_file_path,
                epochs=num_train_epochs,
                gradient_accumulation_steps=1,
            )

            torch.cuda.empty_cache()
            model.to("cpu")


# %%
def plot_alignment_acc(
    seed,
    aligning_layers,
    layers_interval,
    output_dir,
    image_name,
):
    df = pd.DataFrame(columns=["layer", "rel_pos", "accuracy"])
    for rel_pos in range(0, 4):
        for layer in range(0, aligning_layers, layers_interval):
            with open(
                f"{output_dir}/seed.{seed}.rel.{rel_pos}.layer.{layer}/test_log.txt",
                "r",
            ) as file:
                data = file.readlines()
                data = data[1].split("\n")[0]
                df = df.append(
                    {
                        "layer": layer,
                        "rel_pos": rel_pos,
                        "accuracy": float(data),
                    },
                    ignore_index=True,
                )

    df = df.pivot(index="layer", columns="rel_pos", values="accuracy")
    # Reverse the order of the rows and columns in df
    df = df.iloc[::-1]
    df = df.iloc[:, ::-1]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax = sns.heatmap(df, annot=True, fmt=".2f", cmap="rocket_r")
    ax.set_title("IIA for Query Box Identity Variable")
    ax.set_xlabel("Tokens")
    ax.set_ylabel("Layer")
    ax.set_xticklabels([" Box", " ", " X", " contains"])
    ax.set_yticklabels(
        [f"Layer {i}" for i in range(aligning_layers, 0, -layers_interval)], rotation=0
    )
    plt.savefig(f"{image_name}")
    # plt.show()


# %%
def main(args):
    print(args)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=args.model_name_or_path, cache_dir=CACHE_DIR
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load data
    data_sampler = (
        modified_box_name_alignment_example_sampler
        if args.causal_variable == "query_box_identity"
        else object_alignment_example_sampler
    )
    architecture = AutoConfig.from_pretrained(args.model_name_or_path).architectures[0]
    train_dataloader, eval_dataloader, test_dataloader = load_data(
        tokenizer,
        args.data_size,
        data_sampler,
        args.data_file,
        architecture,
        args.num_entities_or_ops,
        args.batch_size,
        args.object_file,
    )

    # Aligning model
    align_model(
        args.aligning_layers,
        args.layers_interval,
        args.model_name_or_path,
        train_dataloader,
        eval_dataloader,
        test_dataloader,
        args.num_train_epochs,
        args.batch_size,
        args.train_log_steps,
        args.seed,
        args.output_dir,
    )

    # Plot alignment accuracy
    plot_alignment_acc(
        args.seed,
        args.aligning_layers,
        args.layers_interval,
        args.output_dir,
        args.image_name,
    )


# %%
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="llama_7b/",
        help="Path to pretrained model",
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default="./box_datasets/no_instructions/alternative/3/train.jsonl",
        help="Path to data file",
    )
    parser.add_argument(
        "--data_size",
        type=int,
        default=3000,
        help="Number of data instances to use",
    )
    parser.add_argument(
        "--num_entities_or_ops",
        type=int,
        default=3,
        help="Number of entities or operations",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results_tmp/",
        help="Path to output directory",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=30,
        help="Batch size for training",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
        help="Number of epochs for training",
    )
    parser.add_argument(
        "--train_log_steps",
        type=int,
        default=3,
        help="Log step for training",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--aligning_layers",
        type=int,
        default=32,
        help="Number of layers to align",
    )
    parser.add_argument(
        "--layers_interval",
        type=int,
        default=1,
        help="Interval between layers to align",
    )
    parser.add_argument(
        "--object_file",
        type=str,
        default="./box_datasets/objects_with_bnc_frequency.csv",
        help="Path to object file",
    )
    parser.add_argument(
        "--causal_variable",
        type=str,
        default="query_box_identity",
        help="Causal variable to align",
    )
    parser.add_argument(
        "--image_name",
        type=str,
        default="alignment_acc.png",
        help="Name of the image to save",
    )

    return parser.parse_args()


# %%
if __name__ == "__main__":
    args = parse_args()
    main(args)
