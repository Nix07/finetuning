import random
import json
import torch
import pandas as pd


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
            output_ids[bi, last_token_indices[bi - 1]] = torch.tensor(labels[bi])

    input_ids = input_tokens["input_ids"].tolist()
    last_token_indices = last_token_indices.tolist()
    output_ids = output_ids.tolist()

    return input_ids, last_token_indices, output_ids


def object_alignment_example_sampler(
    tokenizer, num_samples, data_file, architecture, object_file, num_ents_or_ops=None
):
    input_ids, last_token_indices, output_ids = object_alignment_example_generator(
        tokenizer, num_samples // 2, data_file, object_file
    )

    all_base_input_ids = []
    all_base_input_last_pos = []
    all_source_input_ids = []
    all_source_input_last_pos = []
    all_ctf_output_ids = []
    all_intervention_ids = []

    for i in range(0, num_samples, 2):
        for j in range(2):
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
    output_ids[
        torch.arange(len(last_token_indices)), last_token_indices
    ] = torch.tensor(labels)

    input_ids = input_tokens["input_ids"].tolist()
    last_token_indices = last_token_indices.tolist()
    output_ids = output_ids.tolist()

    return input_ids, last_token_indices, output_ids


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

    for i in range(0, num_samples, num_ents_or_ops):
        if i + num_ents_or_ops > num_samples:
            break
        for j in range(num_ents_or_ops):
            all_base_input_ids += [input_ids[i + j]]
            all_base_input_last_pos += [last_token_indices[i + j]]

            random_source_index = random.choice(
                range(0, num_samples, num_ents_or_ops)
            ) + ((j + 1) % num_ents_or_ops)
            all_source_input_ids += [input_ids[random_source_index]]
            all_source_input_last_pos += [last_token_indices[random_source_index]]

            all_ctf_output_ids += [output_ids[i + (j + 1) % num_ents_or_ops]]
            all_intervention_ids += [0]

    return (
        all_base_input_ids,
        all_base_input_last_pos,
        all_source_input_ids,
        all_source_input_last_pos,
        all_ctf_output_ids,
        all_intervention_ids,
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
    ) = aligner_func(
        tokenizer, data_size, data_file, object_file, num_ents_or_ops, architecture
    )

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
