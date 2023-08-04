import random
import json
import torch
import pandas as pd
import numpy as np


def object_alignment_example_generator(tokenizer, num_samples, data_file, object_file):
    with open(data_file) as f:
        data = [json.loads(line) for line in f]

    objects = pd.read_csv(object_file)

    assert num_samples <= len(data)
    prompts, labels = [], []

    for i in range(num_samples):
        # Example with original object
        prompt = data[i]["sentence"]
        prompts.append(" ".join(prompt.split(" ")[:-1]))
        label = prompt.split(" ")[-1][:-1]
        # 0th index will be BOS token for llama-like tokenizer
        labels.append(tokenizer.encode(label)[1])

        # Example with random object
        # TODO: Assuming no operation instructions
        box_num = prompt.split(". ")[-1].split(" ")[1]
        query = prompt.split(". ")[-1]
        clean_prompt = "".join(prompt.split(". ")[0])
        object = clean_prompt.split(", ")[int(box_num)].split(" ")[-1]
        random_object = random.choice(objects["object_name"].tolist())
        clean_prompt = (
            ", ".join(clean_prompt.split(", ")[: int(box_num)])
            + (", " if int(box_num) != 0 else "")
            + clean_prompt.split(", ")[int(box_num)].replace(object, random_object, 1)
            + (", " if int(box_num) != len(clean_prompt.split(", ")) - 1 else "")
            + ", ".join(clean_prompt.split(", ")[int(box_num) + 1 :])
        )
        prompt = clean_prompt + ". " + query
        prompts.append(" ".join(prompt.split(" ")[:-1]))
        labels.append(tokenizer.encode(random_object)[1])

    input_tokens = tokenizer(prompts, padding=True, return_tensors="pt")
    last_token_indices = input_tokens["attention_mask"].sum(dim=1) - 1
    output_ids = torch.ones_like(input_tokens["input_ids"]) * -100

    for bi in range(len(last_token_indices)):
        if bi % 2 == 0:
            output_ids[bi, last_token_indices[bi]] = torch.tensor(labels[bi])
        else:
            # For random object example, the output should be at the last token index of the original example
            output_ids[bi, last_token_indices[bi - 1]] = torch.tensor(labels[bi])

    input_ids = input_tokens["input_ids"].tolist()
    last_token_indices = last_token_indices.tolist()
    output_ids = output_ids.tolist()

    return input_ids, last_token_indices, output_ids


def object_alignment_example_sampler(
    tokenizer, num_samples, data_file, architecture, object_file, num_ents_or_ops=None
):
    input_ids, last_token_indices, output_ids = object_alignment_example_generator(
        tokenizer, 2 * num_samples, data_file, object_file
    )

    all_base_input_ids = []
    all_base_input_last_pos = []
    all_source_input_ids = []
    all_source_input_last_pos = []
    all_ctf_output_ids = []
    all_intervention_ids = []

    for i in range(0, 2 * num_samples, 2):
        all_base_input_ids += [input_ids[i]]
        all_source_input_ids += [input_ids[i + 1]]
        all_base_input_last_pos += [last_token_indices[i]]
        all_source_input_last_pos += [last_token_indices[i + 1]]
        all_ctf_output_ids += [output_ids[i + 1]]
        all_intervention_ids += [0]

    return (
        all_base_input_ids,
        all_base_input_last_pos,
        all_source_input_ids,
        all_source_input_last_pos,
        all_ctf_output_ids,
        all_intervention_ids,
    )


def entity_tracking_example_sampler(tokenizer, num_samples, data_file, architecture):
    with open(data_file) as f:
        data = [json.loads(line) for line in f]

    assert num_samples <= len(data)
    prompts, labels = [], []

    for i in range(num_samples):
        prompts.append(" ".join(data[i]["sentence"].split(" ")[:-1]))
        label = data[i]["sentence"].split(" ")[-1][:-1]
        # 0th index will be BOS token for llama-like tokenizer
        if architecture in [
            "AlignableLlamaForCausalLM",
            "LLaMAForCausalLM",
            "LlamaForCausalLM",
            "LlaMAForCausalLM",
        ]:
            labels.append(tokenizer.encode(label)[1])
        elif architecture == "GPT2LMHeadModel":
            labels.append(tokenizer.encode(label)[0])
        else:
            raise ValueError(f"Unknown architecture {architecture}")

    input_tokens = tokenizer(prompts, padding=True, return_tensors="pt")
    last_token_indices = input_tokens["attention_mask"].sum(dim=1) - 1
    output_ids = torch.ones_like(input_tokens["input_ids"]) * -100
    output_ids[torch.arange(len(last_token_indices)), last_token_indices] = torch.tensor(labels)

    input_ids = input_tokens["input_ids"].tolist()
    last_token_indices = last_token_indices.tolist()
    output_ids = output_ids.tolist()

    return input_ids, last_token_indices, output_ids


def box_index_aligner_examples(tokenizer, num_samples, data_file, num_ents_or_ops, architecture):
    input_ids, last_token_indices, output_ids = entity_tracking_example_sampler(
        tokenizer, num_samples, data_file, architecture
    )

    all_base_input_ids = []
    all_base_input_last_pos = []
    all_source_input_ids = []
    all_source_input_last_pos = []
    all_ctf_output_ids = []
    all_intervention_ids = []

    for i in range(0, num_samples, num_ents_or_ops):
        for j in range(num_ents_or_ops):
            if i + j >= num_samples:
                break

            all_base_input_ids += [input_ids[i + j]]
            all_base_input_last_pos += [last_token_indices[i + j]]
            all_ctf_output_ids += [output_ids[i + j]]

            random_source_index = random.choice(range(0, num_samples, num_ents_or_ops)) + (
                (j + 1) % num_ents_or_ops
            )
            source_example = input_ids[random_source_index].copy()

            # Randomizing the box indices in the source example
            random_box_indices = np.random.choice(
                list(range(num_ents_or_ops, 10)), size=num_ents_or_ops, replace=False
            )
            random_box = {
                0: tokenizer(str(random_box_indices[0]), return_tensors="pt")
                .input_ids[0, -1]
                .item(),
                1: tokenizer(str(random_box_indices[1]), return_tensors="pt")
                .input_ids[0, -1]
                .item(),
                2: tokenizer(str(random_box_indices[2]), return_tensors="pt")
                .input_ids[0, -1]
                .item(),
            }
            for old_index, new_token in random_box.items():
                old_token = tokenizer(str(old_index), return_tensors="pt").input_ids[0, -1]
                source_example = [
                    new_token if (token == old_token) else token for token in source_example
                ]

            # full_stop_token = 29889
            # full_stop_token_pos = source_example.index(full_stop_token)
            # query_box_index_token = tokenizer(
            #     str((j + 1) % num_ents_or_ops), return_tensors="pt"
            # ).input_ids[0, -1]

            # for pos in range(full_stop_token_pos, len(source_example)):
            #     if source_example[pos] == query_box_index_token:
            #         source_example[pos] = random_box[(j + 1) % num_ents_or_ops]

            all_source_input_ids += [source_example]
            all_source_input_last_pos += [last_token_indices[random_source_index]]

            all_intervention_ids += [0]

    return (
        all_base_input_ids,
        all_base_input_last_pos,
        all_source_input_ids,
        all_source_input_last_pos,
        all_ctf_output_ids,
        all_intervention_ids,
    )


def modified_box_name_alignment_example_sampler(
    tokenizer, num_samples, data_file, object_file, num_ents_or_ops, architecture
):
    input_ids, last_token_indices, output_ids = entity_tracking_example_sampler(
        tokenizer, num_samples, data_file, architecture
    )

    all_base_input_ids = []
    all_base_input_last_pos = []
    all_source_input_ids = []
    all_source_input_last_pos = []
    all_ctf_output_ids = []
    all_intervention_ids = []
    all_incorrect_out_ids = []

    for i in range(0, num_samples, num_ents_or_ops):
        if i + num_ents_or_ops > num_samples:
            break
        for j in range(num_ents_or_ops):
            # Randomize query box index in the base example
            base_example = input_ids[i + j].copy()
            while True:
                # Ensuring the change in position for the query box
                random_pos = np.random.choice(list(range(num_ents_or_ops)))
                if random_pos != j + 1:
                    break
            random_pos_token = (
                tokenizer(str(random_pos), return_tensors="pt").input_ids[0, -1].item()
            )
            query_box_token = (
                tokenizer(str((j + 1) % num_ents_or_ops), return_tensors="pt")
                .input_ids[0, -1]
                .item()
            )
            full_stop_token = 29889
            full_stop_token_index = base_example.index(full_stop_token)
            for ind in range(full_stop_token_index):
                if base_example[ind] == random_pos_token:
                    base_example[ind] = query_box_token
                elif base_example[ind] == query_box_token:
                    base_example[ind] = random_pos_token

            all_base_input_ids += [base_example]
            all_base_input_last_pos += [
                last_token_indices[i + j]
            ]  # Won't change because of randomization
            all_ctf_output_ids += [
                output_ids[i + random_pos]
            ]  # New output will acc. to the new position of the query box
            # all_incorrect_out_ids += [
            #     output_ids[i + j]
            # ]  # Incorrect output will be the original output

            # Choosing a random source example
            random_source_index = random.choice(range(0, num_samples, num_ents_or_ops)) + (
                (j + 1) % num_ents_or_ops
            )
            all_source_input_ids += [input_ids[random_source_index]]
            all_source_input_last_pos += [last_token_indices[random_source_index]]

            all_intervention_ids += [0]

    return (
        all_base_input_ids,
        all_base_input_last_pos,
        all_source_input_ids,
        all_source_input_last_pos,
        all_ctf_output_ids,
        all_intervention_ids,
        all_incorrect_out_ids,
    )


def box_name_alignment_example_sampler(
    tokenizer, num_samples, data_file, architecture, object_file, num_ents_or_ops
):
    input_ids, last_token_indices, output_ids = entity_tracking_example_sampler(
        tokenizer, num_samples, data_file, architecture
    )

    all_base_input_ids = []
    all_base_input_last_pos = []
    all_source_input_ids = []
    all_source_input_last_pos = []
    all_ctf_output_ids = []
    all_intervention_ids = []

    for i in range(0, num_samples, num_ents_or_ops):
        if i + num_ents_or_ops > num_samples:
            break
        for j in range(num_ents_or_ops):
            all_base_input_ids += [input_ids[i]]
            all_source_input_ids += [input_ids[i + j]]
            all_base_input_last_pos += [last_token_indices[i]]
            all_source_input_last_pos += [last_token_indices[i + j]]

            all_ctf_output_ids += [output_ids[i + j]]
            all_intervention_ids += [0]

    return (
        all_base_input_ids,
        all_base_input_last_pos,
        all_source_input_ids,
        all_source_input_last_pos,
        all_ctf_output_ids,
        all_intervention_ids,
    )


def alignment_example_sampler(
    tokenizer,
    data_size,
    aligner_func,
    data_file,
    num_ents_or_ops=None,
    object_file=None,
    architecture=None,
):
    (
        all_base_input_ids,
        all_base_input_last_pos,
        all_source_input_ids,
        all_source_input_last_pos,
        all_ctf_output_ids,
        all_intervention_ids,
    ) = aligner_func(tokenizer, data_size, data_file, object_file, num_ents_or_ops, architecture)

    return (
        all_base_input_ids,
        all_base_input_last_pos,
        all_source_input_ids,
        all_source_input_last_pos,
        all_ctf_output_ids,
        all_intervention_ids,
    )


def factual_sampler(
    tokenizer,
    max_n_training_examples,
    data_file,
    achitecture,
    object_file,
    num_ents_or_ops,
    game,
):
    all_input_ids = []
    all_last_token_indices = []
    all_output_ids = []  # this one does not have input ids, etc..

    if game == "entity_tracking":
        (
            all_input_ids,
            all_last_token_indices,
            all_output_ids,
        ) = object_alignment_example_generator(
            tokenizer,
            max_n_training_examples,
            data_file,
            object_file,
        )

    return all_input_ids, all_last_token_indices, all_output_ids
