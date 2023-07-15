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
from counterfactual_datasets.entity_tracking import (
    box_name_alignment_example_sampler,
    name_alignment_sampler,
)
from trainer import Aligner, CACHE_DIR
import counterfactual_datasets.price_tagging_game as price_tagging_game
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
# Loading dataset

tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path="./llama_7b/", cache_dir=CACHE_DIR
)
tokenizer.pad_token_id = tokenizer.eos_token_id

raw_data = name_alignment_sampler(tokenizer, 3000, box_name_alignment_example_sampler)

# %%
print(f"Toal number of examples: {len(raw_data[0])}")
# %%
# Creating dataset and dataloader

train_size = int(0.8 * len(raw_data[0]))
eval_size = int(0.1 * len(raw_data[0]))
batch_size = 24

raw_train = (
    raw_data[0][:train_size],
    raw_data[1][:train_size],
    raw_data[2][:train_size],
    raw_data[3][:train_size],
    raw_data[4][:train_size],
    raw_data[5][:train_size],
)
raw_eval = (
    raw_data[0][train_size : train_size + eval_size],
    raw_data[1][train_size : train_size + eval_size],
    raw_data[2][train_size : train_size + eval_size],
    raw_data[3][train_size : train_size + eval_size],
    raw_data[4][train_size : train_size + eval_size],
    raw_data[5][train_size : train_size + eval_size],
)
raw_test = (
    raw_data[0][train_size + eval_size :],
    raw_data[1][train_size + eval_size :],
    raw_data[2][train_size + eval_size :],
    raw_data[3][train_size + eval_size :],
    raw_data[4][train_size + eval_size :],
    raw_data[5][train_size + eval_size :],
)
train_dataset = Dataset.from_dict(
    {
        "base_input_ids": raw_train[0],
        "base_input_last_pos": raw_train[1],
        "source_input_ids": raw_train[2],
        "source_input_last_pos": raw_train[3],
        "labels": raw_train[4],
        "intervention_ids": raw_train[5],
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
        "intervention_ids": raw_eval[5],
    }
).with_format("torch")
eval_dataloader = DataLoader(
    eval_dataset,
    batch_size=batch_size,
)
test_dataset = Dataset.from_dict(
    {
        "base_input_ids": raw_eval[0],
        "base_input_last_pos": raw_eval[1],
        "source_input_ids": raw_eval[2],
        "source_input_last_pos": raw_eval[3],
        "labels": raw_eval[4],
        "intervention_ids": raw_eval[5],
    }
).with_format("torch")
test_dataloader = DataLoader(
    test_dataset,
    batch_size=batch_size,
)


# %%
# Loading updated model with rotation matrix parameters

for rel_pos in range(0, 4):
    for layer in range(0, 32, 4):
        print(f"Starting traing for layer: {layer}")
        alignment_config = {
            "layer": layer,
            "rel_pos": rel_pos,
        }
        model = AutoAlignableModel.from_pretrained(
            "./llama_7b/",
            alignment_config=alignment_config,
            torch_dtype=torch.float32,
        )
        _ = model.to("cuda")  # first GPU

        # Training configuration

        # set off the gradients among all other layers.
        for name, param in model.named_parameters():
            if "rotate_layer" not in name and "intervention_boundaries" not in name:
                param.requires_grad = False
            else:
                logger.info(f"Requiring gradients on layer: {name}")
        t_total = int(len(train_dataloader) * 3)
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

        # You can define your custom compute_metrics function.
        def compute_metrics(eval_preds, eval_labels):
            # eval_preds: (#batch, vocab_size)
            # eval_labels: (#batch)
            total_count = 0
            correct_count = 0
            for eval_pred, eval_label in zip(eval_preds, eval_labels):
                actual_test_labels = eval_label
                pred_test_labels = torch.argmax(eval_pred, dim=-1)
                correct_labels = actual_test_labels == pred_test_labels
                total_count += 1
                correct_count += correct_labels.sum().tolist()
            accuracy = round(correct_count / total_count, 2)
            return {"accuracy": accuracy}

        model_type = AutoConfig.from_pretrained("./llama_7b/").architectures[0]

        run_name = f"{model_type}.seed.42.{alignment_config['rel_pos']}.{alignment_config['layer']}"
        if not os.path.exists("./results_tmp/"):
            os.mkdir("./results_tmp/")
        os.environ["WANDB_PROJECT"] = f"Boundless-DAS"
        output_dir = os.path.join("./results_tmp/", run_name)
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

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
            log_step=10,
            valid_steps=50,
            output_dir=output_dir,
            epochs=1,
            gradient_accumulation_steps=4,
        )

        torch.cuda.empty_cache()
        model.to("cpu")

# %%
df = pd.DataFrame(columns=["layer", "rel_pos", "accuracy"])
for rel_pos in range(0, 4):
    for layer in range(0, 32, 4):
        with open(
            f"results_tmp/LlamaForCausalLM.seed.42.{rel_pos}.{layer}/eval_log.txt",
            "r",
        ) as file:
            data = file.readlines()
            data = data[-1].split(",")[1]
            df = df.append(
                {
                    "layer": layer,
                    "rel_pos": rel_pos,
                    "accuracy": float(data),
                },
                ignore_index=True,
            )

# %%
df = df.pivot(index="layer", columns="rel_pos", values="accuracy")

# %%
# Reverse the order of the rows and columns in df
df = df.iloc[::-1]
df = df.iloc[:, ::-1]

# %%
ax = sns.heatmap(df, annot=True, fmt=".2f", cmap="rocket_r")
ax.set_title("IIA for Query Box Identity Variable")
ax.set_xlabel("Tokens")
ax.set_ylabel("Layer")
ax.set_xticklabels([" Box", " ", " X", " contains"])
ax.set_yticklabels([f"Layer {i}" for i in range(32, -1, -4)], rotation=0)
plt.savefig("query-box-var.png")
plt.show()
# %%

checkpoint_state_dict_best = torch.load(
    f"results_tmp/LlamaForCausalLM.seed.42.0.15/pytorch-rotate-best.bin"
)
# %%
