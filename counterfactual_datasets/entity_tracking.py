import random
import json
import torch
import pandas as pd
import numpy as np


def change_box_label(
    tokenizer,
    num_samples,
    data_file,
    object_file,
    num_ents_or_ops,
    architecture,
    few_shot,
    alt_examples,
):
    with open(data_file) as f:
        data = [json.loads(line) for line in f]

    assert num_samples <= len(data)
    prompts, labels = [], []

    for i in range(num_samples):
        org_prompt = " ".join(data[i]["sentence"].split(" ")[:-1])
        prompts.append(org_prompt)
        label = data[i]["sentence"].split(" ")[-1][:-1]
        labels.append(tokenizer.encode(label)[1])

        new_prompt = org_prompt.replace(" 0", " 6")
        new_prompt = new_prompt.replace(" 1", " 4")
        new_prompt = new_prompt.replace(" 2", " 9")
        prompts.append(new_prompt)
        labels.append(tokenizer.encode(label)[1])

    input_tokens = tokenizer(prompts, padding=True, return_tensors="pt")
    last_token_indices = input_tokens["attention_mask"].sum(dim=1) - 1
    input_ids = input_tokens["input_ids"].tolist()
    last_token_indices = last_token_indices.tolist()
    output_ids = labels

    return input_ids, last_token_indices, output_ids


def shift_box_positions(
    tokenizer, num_samples, data_file, object_file, num_boxes, few_shot, alt_examples
):
    with open(data_file) as f:
        data = [json.loads(line) for line in f]

    assert num_samples <= len(data)
    prompts, labels = [], []

    for i in range(num_samples):
        org_prompt = " ".join(data[i]["sentence"].split(" ")[:-1])
        prompts.append(org_prompt)
        label = data[i]["sentence"].split(" ")[-1][:-1]
        labels.append(tokenizer.encode(label)[1])

        query = org_prompt.split(". ")[-1]
        clean_prompt = org_prompt.split(". ")[0]
        clean_prompt = clean_prompt.split(", ")
        new_prompt = []
        for seg_idx in range(len(clean_prompt)):
            new_prompt.append(clean_prompt[(seg_idx + 1) % num_boxes])
        new_prompt = ", ".join(new_prompt)
        prompts.append(new_prompt + ". " + query)
        labels.append(tokenizer.encode(label)[1])

    input_tokens = tokenizer(prompts, padding=True, return_tensors="pt")
    last_token_indices = input_tokens["attention_mask"].sum(dim=1) - 1
    input_ids = input_tokens["input_ids"].tolist()
    last_token_indices = last_token_indices.tolist()
    output_ids = labels

    return input_ids, last_token_indices, output_ids


def object_alignment_example_generator(
    tokenizer, num_samples, data_file, object_file, few_shot, alt_examples
):
    with open(data_file) as f:
        data = [json.loads(line) for line in f]

    objects = pd.read_csv(object_file)

    assert num_samples <= len(data)
    prompts, all_object_tokens, labels = [], [], []

    if alt_examples:
        priminig_examples = """Watch is in Box 0, nothing is in Box 1, bottle is in Box 2. Box 2 contains bottle.\n Wire is in Box 0, biscotti is in Box 1, camera is in Box 2. Box 1 contains biscotti.\n Nothing is in Box 0, tetrapod is in Box 1, incense is in Box 2. Box 0 contains nothing.\n """
    else:
        priminig_examples = ""

    for i in range(num_samples):
        # Example with original object
        label = data[i]["sentence"].split(" ")[-1][:-1]
        # 0th index will be BOS token for llama-like tokenizer
        labels.append(tokenizer.encode(label)[1])

        object_index_in_segment = 0 if alt_examples else 3
        all_objects = [
            segment.split(" ")[object_index_in_segment].lower()
            for segment in data[i]["sentence"].split(".")[0].split(", ")
        ]
        all_object_tokens.append([tokenizer.encode(obj)[1] for obj in all_objects])

        org_prompt = " ".join(data[i]["sentence"].split(" ")[:-1])
        prompt = priminig_examples + org_prompt if few_shot else org_prompt
        prompts.append(prompt)

        # Example with random object
        box_num = org_prompt.split(". ")[-1].split(" ")[1]
        query = org_prompt.split(". ")[-1]
        clean_prompt = org_prompt.split(". ")[0]
        object = (
            clean_prompt.split(", ")[int(box_num)].split(" ")[0]
            if alt_examples
            else clean_prompt.split(", ")[int(box_num)].split(" ")[-1]
        )
        random_object = random.choice(objects["object_name"].tolist())

        # Capitalizing the first letter of the object
        if alt_examples and int(box_num) == 0:
            random_object = random_object[0].upper() + random_object[1:]

        clean_prompt = (
            ", ".join(clean_prompt.split(", ")[: int(box_num)])
            + (", " if int(box_num) != 0 else "")
            + clean_prompt.split(", ")[int(box_num)].replace(object, random_object, 1)
            + (", " if int(box_num) != len(clean_prompt.split(", ")) - 1 else "")
            + ", ".join(clean_prompt.split(", ")[int(box_num) + 1 :])
        )

        prompt = (
            priminig_examples + clean_prompt + ". " + query
            if few_shot
            else clean_prompt + ". " + query
        )

        prompts.append(prompt)
        labels.append(tokenizer.encode(random_object.lower())[1])
        all_objects = [random_object if obj == object else obj for obj in all_objects]
        all_object_tokens.append([tokenizer.encode(obj)[1] for obj in all_objects])

    input_tokens = tokenizer(prompts, padding=True, return_tensors="pt")
    last_token_indices = input_tokens["attention_mask"].sum(dim=1) - 1
    output_ids = torch.tensor(labels)

    # output_ids = torch.ones_like(input_tokens["input_ids"]) * -100

    # for bi in range(len(last_token_indices)):
    #     if bi % 2 == 0:
    #         output_ids[bi, last_token_indices[bi]] = torch.tensor(labels[bi])
    #     else:
    #         # For random object example, the output should be at the last token index of the original example
    #         output_ids[bi, last_token_indices[bi - 1]] = torch.tensor(labels[bi])

    input_ids = input_tokens["input_ids"].tolist()
    last_token_indices = last_token_indices.tolist()
    output_ids = output_ids.tolist()

    return input_ids, last_token_indices, output_ids, all_object_tokens


def change_query_box_pos(
    tokenizer,
    num_samples,
    data_file,
    object_file,
    num_ents_or_ops,
    architecture,
    few_shot,
    alt_examples,
):
    input_ids, last_token_indices, output_ids = change_box_label(
        tokenizer,
        num_samples,
        data_file,
        object_file,
        num_ents_or_ops,
        architecture,
        few_shot,
        alt_examples,
    )

    all_base_input_ids = []
    all_base_input_last_pos = []
    all_source_input_ids = []
    all_source_input_last_pos = []
    all_ctf_output_ids = []

    for i in range(0, num_samples, 2):
        all_base_input_ids += [input_ids[i]]
        all_base_input_last_pos += [last_token_indices[i]]
        all_source_input_ids += [input_ids[i + 1]]
        all_source_input_last_pos += [last_token_indices[i + 1]]
        all_ctf_output_ids += [output_ids[i]]

    return (
        all_base_input_ids,
        all_base_input_last_pos,
        all_source_input_ids,
        all_source_input_last_pos,
        all_ctf_output_ids,
    )


def shift_query_position_example_sampler(
    tokenizer,
    num_samples,
    data_file,
    object_file,
    num_ents_or_ops,
    architecture,
    few_shot,
    alt_examples,
):
    input_ids, last_token_indices, output_ids = shift_box_positions(
        tokenizer,
        num_samples,
        data_file,
        object_file,
        num_ents_or_ops,
        few_shot,
        alt_examples,
    )

    all_base_input_ids = []
    all_base_input_last_pos = []
    all_source_input_ids = []
    all_source_input_last_pos = []
    all_ctf_output_ids = []

    for i in range(0, num_samples, 2):
        all_base_input_ids += [input_ids[i]]
        all_base_input_last_pos += [last_token_indices[i]]
        all_source_input_ids += [input_ids[i + 1]]
        all_source_input_last_pos += [last_token_indices[i + 1]]
        all_ctf_output_ids += [output_ids[i]]

    return (
        all_base_input_ids,
        all_base_input_last_pos,
        all_source_input_ids,
        all_source_input_last_pos,
        all_ctf_output_ids,
    )


def object_alignment_example_sampler(
    tokenizer,
    num_samples,
    data_file,
    architecture,
    object_file,
    num_ents_or_ops,
    few_shot,
    alt_examples,
):
    num_samples = 2 * num_samples
    (
        input_ids,
        last_token_indices,
        output_ids,
        object_ids,
    ) = object_alignment_example_generator(
        tokenizer, num_samples, data_file, object_file, few_shot, alt_examples
    )

    all_base_input_ids = []
    all_base_input_last_pos = []
    all_source_input_ids = []
    all_source_input_last_pos = []
    all_ctf_output_ids = []
    all_object_ids = []
    all_intervention_ids = []

    for i in range(0, num_samples, 2):
        all_base_input_ids += [input_ids[i]]
        all_source_input_ids += [input_ids[i + 1]]
        all_base_input_last_pos += [last_token_indices[i]]
        all_source_input_last_pos += [last_token_indices[i + 1]]
        all_ctf_output_ids += [output_ids[i + 1]]
        all_object_ids += [object_ids[i]]
        all_intervention_ids += [0]

    return (
        all_base_input_ids,
        all_base_input_last_pos,
        all_source_input_ids,
        all_source_input_last_pos,
        all_ctf_output_ids,
        all_object_ids,
        all_intervention_ids,
    )


def entity_tracking_example_sampler(
    tokenizer, num_samples, data_file, architecture, few_shot, alt_examples
):
    with open(data_file) as f:
        data = [json.loads(line) for line in f]

    assert num_samples <= len(data)
    prompts, incorrect_object_tokens, labels = [], [], []

    # if alt_examples:
    #     priminig_examples = """Watch is in Box 0, nothing is in Box 1, bottle is in Box 2. Box 2 contains bottle.\n Wire is in Box 0, biscotti is in Box 1, camera is in Box 2. Box 1 contains biscotti.\n Nothing is in Box 0, tetrapod is in Box 1, incense is in Box 2. Box 0 contains nothing.\n """
    # else:
    #     priminig_examples = ""

    for i in range(num_samples):
        label = data[i]["sentence"].split(" ")[-1][:-1]
        object_index_in_segment = 1 if alt_examples else 4
        incorrect_objects = [
            segment.split(" ")[object_index_in_segment].lower()
            for segment in data[i]["sentence"].split(".")[0].split(", ")
        ]
        incorrect_objects.remove(label.lower())
        incorrect_object_tokens.append(
            [tokenizer.encode(obj)[1] for obj in incorrect_objects]
        )

        prompt = " ".join(data[i]["sentence"].split(" ")[:-1])
        # if few_shot:
        #     prompt = priminig_examples + prompt

        prompts.append(prompt)

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
    # output_ids = torch.ones_like(input_tokens["input_ids"]) * -100
    # output_ids[torch.arange(len(last_token_indices)), last_token_indices] = torch.tensor(labels)
    output_ids = torch.tensor(labels)

    input_ids = input_tokens["input_ids"].tolist()
    last_token_indices = last_token_indices.tolist()
    output_ids = output_ids.tolist()

    return input_ids, last_token_indices, output_ids


def random_label_samples_for_path_patching(
    tokenizer,
    num_samples,
    data_file,
    num_ents_or_ops,
    architecture,
    few_shot,
    alt_examples,
):
    input_ids, last_token_indices, output_ids = entity_tracking_example_sampler(
        tokenizer, num_samples, data_file, architecture, few_shot, alt_examples
    )

    all_base_input_ids = []
    all_base_input_last_pos = []
    all_source_input_ids = []
    all_source_input_last_pos = []
    all_ctf_output_ids = []

    for i in range(0, num_samples, num_ents_or_ops):
        for j in range(num_ents_or_ops):
            if i + j >= num_samples:
                break

            all_base_input_ids += [input_ids[i + j]]
            all_base_input_last_pos += [last_token_indices[i + j]]
            all_ctf_output_ids += [output_ids[i + j]]

            random_source_index = random.choice(
                list(range(0, num_samples, num_ents_or_ops))
            )
            random_source_index += (j + 1) % num_ents_or_ops
            all_source_input_ids += [input_ids[random_source_index]]
            all_source_input_last_pos += [last_token_indices[random_source_index]]

    return (
        all_base_input_ids,
        all_base_input_last_pos,
        all_source_input_ids,
        all_source_input_last_pos,
        all_ctf_output_ids,
    )


def box_index_aligner_examples(
    tokenizer,
    num_samples,
    data_file,
    num_ents_or_ops,
    architecture,
    few_shot,
    alt_examples,
):
    (
        input_ids,
        last_token_indices,
        output_ids,
    ) = entity_tracking_example_sampler(
        tokenizer, num_samples, data_file, architecture, few_shot, alt_examples
    )

    all_base_input_ids = []
    all_base_input_last_pos = []
    all_source_input_ids = []
    all_source_input_last_pos = []
    all_ctf_output_ids = []
    all_intervention_ids = []
    all_incorrect_output_ids = []

    for i in range(0, num_samples, num_ents_or_ops):
        for j in range(num_ents_or_ops):
            if i + j >= num_samples:
                break

            all_base_input_ids += [input_ids[i + j]]
            all_base_input_last_pos += [last_token_indices[i + j]]
            all_ctf_output_ids += [output_ids[i + j]]

            temp = []
            for ind in range(1, num_ents_or_ops):
                temp += [output_ids[i + ((j + ind) % num_ents_or_ops)]]
            all_incorrect_output_ids += [temp]

            random_source_index = random.choice(
                list(range(0, num_samples, num_ents_or_ops))
            )
            random_source_index += (j + 1) % num_ents_or_ops
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
                old_token = tokenizer(str(old_index), return_tensors="pt").input_ids[
                    0, -1
                ]
                source_example = [
                    new_token if (token == old_token) else token
                    for token in source_example
                ]

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
        all_incorrect_output_ids,
    )


def modified_box_name_alignment_example_sampler(
    tokenizer,
    num_samples,
    data_file,
    object_file,
    num_ents_or_ops,
    architecture,
    few_shot,
    alt_examples,
):
    (
        input_ids,
        last_token_indices,
        output_ids,
        incorrect_object_ids,
    ) = entity_tracking_example_sampler(
        tokenizer, num_samples, data_file, architecture, few_shot, alt_examples
    )

    all_base_input_ids = []
    all_base_input_last_pos = []
    all_source_input_ids = []
    all_source_input_last_pos = []
    all_ctf_output_ids = []
    all_exp_objects = []
    all_intervention_ids = []

    for i in range(0, num_samples, num_ents_or_ops):
        if i + num_ents_or_ops > num_samples:
            break
        for j in range(num_ents_or_ops):
            all_base_input_ids += [input_ids[i + j]]
            all_base_input_last_pos += [last_token_indices[i + j]]
            all_exp_objects += [incorrect_object_ids[i + j]]

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
        all_exp_objects,
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
    few_shot=False,
    alt_examples=False,
):
    (
        all_base_input_ids,
        all_base_input_last_pos,
        all_source_input_ids,
        all_source_input_last_pos,
        all_ctf_output_ids,
        all_incorrect_object_ids,
        all_intervention_ids,
    ) = aligner_func(
        tokenizer=tokenizer,
        num_samples=data_size,
        data_file=data_file,
        object_file=object_file,
        num_ents_or_ops=num_ents_or_ops,
        architecture=architecture,
        few_shot=few_shot,
        alt_examples=alt_examples,
    )

    return (
        all_base_input_ids,
        all_base_input_last_pos,
        all_source_input_ids,
        all_source_input_last_pos,
        all_ctf_output_ids,
        all_incorrect_object_ids,
        all_intervention_ids,
    )
