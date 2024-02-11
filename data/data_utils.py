import random
import json
import torch
import pandas as pd
import numpy as np


def get_data_for_mean_ablation(
    tokenizer,
    num_samples,
    data_file,
    num_boxes,
):
    """
    This function returns the data for the mean ablation experiment,
    which consists of examples with different set of objects, box
    labels and randomly selected query box label.

    Args:
        tokenizer (transformers.tokenizer): Tokenizer object
        num_samples (int): Number of samples to generate
        data_file (str): Path to the data file
        num_boxes (int): Number of boxes in the scene
    """

    with open(data_file, encoding="utf-8") as file_handle:
        data = [json.loads(line) for line in file_handle]

    assert num_samples <= len(data)
    prompts = []

    # Each prompt will have different set of objects and box labels,
    # with random query box label
    for i in range(0, num_samples, num_boxes):
        prompt = " ".join(data[i]["sentence"].split(" ")[:-1])
        prompt_query = prompt.split(". ", maxsplit=1)[-1]
        random_alphabet = chr(random.randint(65, 90))
        prompt_query = (
            prompt_query.split(" ")[0]
            + " "
            + random_alphabet
            + " "
            + " ".join(prompt_query.split(" ")[2:])
        )
        prompt = prompt.split(". ", maxsplit=1)[0] + ". " + prompt_query
        prompts.append(prompt)

    input_tokens = tokenizer(prompts, padding=True, return_tensors="pt")
    last_token_indices = input_tokens["attention_mask"].sum(dim=1) - 1

    return (
        input_tokens["input_ids"],
        last_token_indices,
    )


def alter_box_object_association(
    tokenizer,
    num_samples,
    data_file,
    object_file,
    num_boxes,
    alt_format,
    correct_pred_indices=[],
):
    with open(data_file) as f:
        data = [json.loads(line) for line in f]

    assert num_samples <= len(data)
    base_prompts, source_prompts, labels = [], [], []

    for i in range(0, num_samples):
        base_prompt = " ".join(data[i]["sentence"].split(" ")[:-1])
        base_query = base_prompt.split(". ")[-1]
        base_query_box_label = base_query.split(" ")[1]
        base_query_box_pos = [
            idx
            for idx, segment in enumerate(base_query.split(". ")[0].split(", "))
            if base_query_box_label in segment
        ][0]
        if base_query_box_pos == -1:
            raise ValueError("Box label not found in the base prompt")
        base_prompts.append(base_prompt)

        source_query_box_pos = base_query_box_pos
        random_choices = list(range(0, num_samples))
        random.shuffle(random_choices)
        while source_query_box_pos == base_query_box_pos:
            random_source_index = random.choice(random_choices)
            source_prompt = " ".join(
                data[random_source_index]["sentence"].split(" ")[:-1]
            )
            source_query = source_prompt.split(". ")[-1]
            source_query_box_label = source_query.split(" ")[1]
            source_query_box_pos = [
                idx
                for idx, segment in enumerate(source_prompt.split(". ")[0].split(", "))
                if source_query_box_label in segment
            ][0]
            random_choices.remove(random_source_index)

        source_segment = source_prompt.split(". ")[0].split(", ")[source_query_box_pos]
        source_segment = source_segment.replace(" is in", " is not in")
        source_prompt = (
            ", ".join(source_prompt.split(". ")[0].split(", ")[:source_query_box_pos])
            + (", " if source_query_box_pos != 0 else "")
            + source_segment
            + (
                ", "
                if source_query_box_pos
                != len(source_prompt.split(". ")[0].split(", ")) - 1
                else ""
            )
            + ", ".join(
                source_prompt.split(". ")[0].split(", ")[source_query_box_pos + 1 :]
            )
            + ". "
            + source_prompt.split(". ")[-1]
        )
        source_prompts.append(source_prompt)

        base_prompt = base_prompt.split(". ")[0]
        correct_object = base_prompt.split(", ")[source_query_box_pos].split(" ")[1]
        labels.append(tokenizer.encode(correct_object)[1])

    base_input_tokens = tokenizer(base_prompts, padding=True, return_tensors="pt")[
        "input_ids"
    ]
    base_last_token_indices = (
        tokenizer(base_prompts, padding=True, return_tensors="pt")[
            "attention_mask"
        ].sum(dim=1)
        - 1
    )
    source_input_tokens = tokenizer(source_prompts, padding=True, return_tensors="pt")[
        "input_ids"
    ]
    source_last_token_indices = (
        tokenizer(source_prompts, padding=True, return_tensors="pt")[
            "attention_mask"
        ].sum(dim=1)
        - 1
    )
    output_ids = torch.tensor(labels)

    return (
        base_input_tokens,
        base_last_token_indices,
        source_input_tokens,
        source_last_token_indices,
        output_ids,
    )


def add_box_before_correct_segment(
    tokenizer,
    num_samples,
    data_file,
    object_file,
    num_boxes,
    alt_format,
    correct_pred_indices=[],
):
    with open(data_file) as f:
        data = [json.loads(line) for line in f]

    assert num_samples <= len(data)
    base_prompts, source_prompts, labels = [], [], []

    for i in range(0, num_samples):
        base_prompt = " ".join(data[i]["sentence"].split(" ")[:-1])
        base_query = base_prompt.split(". ")[-1]
        base_query_box_label = base_query.split(" ")[1]
        base_query_box_pos = [
            idx
            for idx, segment in enumerate(base_query.split(". ")[0].split(", "))
            if base_query_box_label in segment
        ][0]
        if base_query_box_pos == -1:
            raise ValueError("Box label not found in the base prompt")
        base_prompts.append(base_prompt)

        source_query_box_pos = base_query_box_pos
        random_choices = list(range(0, num_samples))
        random.shuffle(random_choices)
        while source_query_box_pos == base_query_box_pos:
            random_source_index = random.choice(random_choices)
            source_prompt = " ".join(
                data[random_source_index]["sentence"].split(" ")[:-1]
            )
            source_query = source_prompt.split(". ")[-1]
            source_query_box_label = source_query.split(" ")[1]
            source_query_box_pos = [
                idx
                for idx, segment in enumerate(source_prompt.split(". ")[0].split(", "))
                if source_query_box_label in segment
            ][0]
            random_choices.remove(random_source_index)

        source_segment = source_prompt.split(". ")[0].split(", ")[source_query_box_pos]
        if source_query_box_pos != 0:
            source_segment = (
                "there are three additional boxes, Box PP, Box BB and Box AA, "
                + source_segment
            )
        else:
            source_segment = (
                "There are three additional boxes, Box PP, Box BB and Box AA, "
                + source_segment
            )
        source_prompt = (
            ", ".join(source_prompt.split(". ")[0].split(", ")[:source_query_box_pos])
            + (", " if source_query_box_pos != 0 else "")
            + source_segment
            + (
                ", "
                if source_query_box_pos
                != len(source_prompt.split(". ")[0].split(", ")) - 1
                else ""
            )
            + ", ".join(
                source_prompt.split(". ")[0].split(", ")[source_query_box_pos + 1 :]
            )
            + ". "
            + source_prompt.split(". ")[-1]
        )
        source_prompts.append(source_prompt)

        base_prompt = base_prompt.split(". ")[0]
        correct_object = base_prompt.split(", ")[source_query_box_pos].split(" ")[1]
        labels.append(tokenizer.encode(correct_object)[1])

    base_input_tokens = tokenizer(base_prompts, padding=True, return_tensors="pt")[
        "input_ids"
    ]
    base_last_token_indices = (
        tokenizer(base_prompts, padding=True, return_tensors="pt")[
            "attention_mask"
        ].sum(dim=1)
        - 1
    )
    source_input_tokens = tokenizer(source_prompts, padding=True, return_tensors="pt")[
        "input_ids"
    ]
    source_last_token_indices = (
        tokenizer(source_prompts, padding=True, return_tensors="pt")[
            "attention_mask"
        ].sum(dim=1)
        - 1
    )
    output_ids = torch.tensor(labels)

    return (
        base_input_tokens,
        base_last_token_indices,
        source_input_tokens,
        source_last_token_indices,
        output_ids,
    )


def add_raw_text_at_end(
    tokenizer,
    num_samples,
    data_file,
    object_file,
    num_boxes,
    alt_format,
    correct_pred_indices=[],
):
    with open(data_file) as f:
        data = [json.loads(line) for line in f]

    assert num_samples <= len(data)
    base_prompts, source_prompts, labels = [], [], []

    for i in range(0, num_samples):
        base_prompt = " ".join(data[i]["sentence"].split(" ")[:-1])
        base_query = base_prompt.split(". ")[-1]
        base_query_box_label = base_query.split(" ")[1]
        base_query_box_pos = [
            idx
            for idx, segment in enumerate(base_query.split(". ")[0].split(", "))
            if base_query_box_label in segment
        ][0]
        if base_query_box_pos == -1:
            raise ValueError("Box label not found in the base prompt")
        base_prompts.append(base_prompt)

        source_query_box_pos = base_query_box_pos
        random_choices = list(range(0, num_samples))
        random.shuffle(random_choices)
        while source_query_box_pos == base_query_box_pos:
            random_source_index = random.choice(random_choices)
            source_prompt = " ".join(
                data[random_source_index]["sentence"].split(" ")[:-1]
            )
            source_query = source_prompt.split(". ")[-1]
            source_query_box_label = source_query.split(" ")[1]
            source_query_box_pos = [
                idx
                for idx, segment in enumerate(source_prompt.split(". ")[0].split(", "))
                if source_query_box_label in segment
            ][0]
            random_choices.remove(random_source_index)

        source_prompt = (
            source_prompt.split(". ")[0]
            + ", these are a bunch of boxes containing objects"
            + ". "
            + source_prompt.split(". ")[-1]
        )
        source_prompts.append(source_prompt)

        base_prompt = base_prompt.split(". ")[0]
        correct_object = base_prompt.split(", ")[source_query_box_pos].split(" ")[1]
        labels.append(tokenizer.encode(correct_object)[1])

    base_input_tokens = tokenizer(base_prompts, padding=True, return_tensors="pt")[
        "input_ids"
    ]
    base_last_token_indices = (
        tokenizer(base_prompts, padding=True, return_tensors="pt")[
            "attention_mask"
        ].sum(dim=1)
        - 1
    )
    source_input_tokens = tokenizer(source_prompts, padding=True, return_tensors="pt")[
        "input_ids"
    ]
    source_last_token_indices = (
        tokenizer(source_prompts, padding=True, return_tensors="pt")[
            "attention_mask"
        ].sum(dim=1)
        - 1
    )
    output_ids = torch.tensor(labels)

    return (
        base_input_tokens,
        base_last_token_indices,
        source_input_tokens,
        source_last_token_indices,
        output_ids,
    )


def add_raw_text_at_start(
    tokenizer,
    num_samples,
    data_file,
    object_file,
    num_boxes,
    alt_format,
    correct_pred_indices=[],
):
    with open(data_file) as f:
        data = [json.loads(line) for line in f]

    assert num_samples <= len(data)
    base_prompts, source_prompts, labels = [], [], []

    for i in range(0, num_samples):
        base_prompt = " ".join(data[i]["sentence"].split(" ")[:-1])
        base_query = base_prompt.split(". ")[-1]
        base_query_box_label = base_query.split(" ")[1]
        base_query_box_pos = [
            idx
            for idx, segment in enumerate(base_query.split(". ")[0].split(", "))
            if base_query_box_label in segment
        ][0]
        if base_query_box_pos == -1:
            raise ValueError("Box label not found in the base prompt")
        base_prompts.append(base_prompt)

        source_query_box_pos = base_query_box_pos
        random_choices = list(range(0, num_samples))
        random.shuffle(random_choices)
        while source_query_box_pos == base_query_box_pos:
            random_source_index = random.choice(random_choices)
            source_prompt = " ".join(
                data[random_source_index]["sentence"].split(" ")[:-1]
            )
            source_query = source_prompt.split(". ")[-1]
            source_query_box_label = source_query.split(" ")[1]
            source_query_box_pos = [
                idx
                for idx, segment in enumerate(source_prompt.split(". ")[0].split(", "))
                if source_query_box_label in segment
            ][0]
            random_choices.remove(random_source_index)

        source_prompt = (
            "There are a bunch of boxes containing objects, "
            + source_prompt[0].lower()
            + source_prompt[1:]
        )
        source_prompts.append(source_prompt)

        base_prompt = base_prompt.split(". ")[0]
        correct_object = base_prompt.split(", ")[source_query_box_pos].split(" ")[1]
        labels.append(tokenizer.encode(correct_object)[1])

    base_input_tokens = tokenizer(base_prompts, padding=True, return_tensors="pt")[
        "input_ids"
    ]
    base_last_token_indices = (
        tokenizer(base_prompts, padding=True, return_tensors="pt")[
            "attention_mask"
        ].sum(dim=1)
        - 1
    )
    source_input_tokens = tokenizer(source_prompts, padding=True, return_tensors="pt")[
        "input_ids"
    ]
    source_last_token_indices = (
        tokenizer(source_prompts, padding=True, return_tensors="pt")[
            "attention_mask"
        ].sum(dim=1)
        - 1
    )
    output_ids = torch.tensor(labels)

    return (
        base_input_tokens,
        base_last_token_indices,
        source_input_tokens,
        source_last_token_indices,
        output_ids,
    )


def add_segment_at_end(
    tokenizer,
    num_samples,
    data_file,
    object_file,
    num_boxes,
    alt_format,
    correct_pred_indices=[],
):
    with open(data_file) as f:
        data = [json.loads(line) for line in f]

    assert num_samples <= len(data)
    base_prompts, source_prompts, labels = [], [], []

    for i in range(0, num_samples):
        base_prompt = " ".join(data[i]["sentence"].split(" ")[:-1])
        base_query = base_prompt.split(". ")[-1]
        base_query_box_label = base_query.split(" ")[1]
        base_query_box_pos = [
            idx
            for idx, segment in enumerate(base_query.split(". ")[0].split(", "))
            if base_query_box_label in segment
        ][0]
        if base_query_box_pos == -1:
            raise ValueError("Box label not found in the base prompt")
        base_prompts.append(base_prompt)

        source_query_box_pos = base_query_box_pos
        random_choices = list(range(0, num_samples))
        random.shuffle(random_choices)
        while source_query_box_pos == base_query_box_pos:
            random_source_index = random.choice(random_choices)
            source_prompt = " ".join(
                data[random_source_index]["sentence"].split(" ")[:-1]
            )
            source_query = source_prompt.split(". ")[-1]
            source_query_box_label = source_query.split(" ")[1]
            source_query_box_pos = [
                idx
                for idx, segment in enumerate(source_prompt.split(". ")[0].split(", "))
                if source_query_box_label in segment
            ][0]
            random_choices.remove(random_source_index)

        source_prompt = (
            source_prompt.split(". ")[0]
            + ", the apple is in Box O"
            + ". "
            + source_prompt.split(". ")[-1]
        )
        source_prompts.append(source_prompt)

        base_prompt = base_prompt.split(". ")[0]
        correct_object = base_prompt.split(", ")[source_query_box_pos].split(" ")[1]
        labels.append(tokenizer.encode(correct_object)[1])

    base_input_tokens = tokenizer(base_prompts, padding=True, return_tensors="pt")[
        "input_ids"
    ]
    base_last_token_indices = (
        tokenizer(base_prompts, padding=True, return_tensors="pt")[
            "attention_mask"
        ].sum(dim=1)
        - 1
    )
    source_input_tokens = tokenizer(source_prompts, padding=True, return_tensors="pt")[
        "input_ids"
    ]
    source_last_token_indices = (
        tokenizer(source_prompts, padding=True, return_tensors="pt")[
            "attention_mask"
        ].sum(dim=1)
        - 1
    )
    output_ids = torch.tensor(labels)

    return (
        base_input_tokens,
        base_last_token_indices,
        source_input_tokens,
        source_last_token_indices,
        output_ids,
    )


def add_segment_at_start(
    tokenizer,
    num_samples,
    data_file,
    object_file,
    num_boxes,
    alt_format,
    correct_pred_indices=[],
):
    with open(data_file) as f:
        data = [json.loads(line) for line in f]

    assert num_samples <= len(data)
    base_prompts, source_prompts, labels = [], [], []

    for i in range(0, num_samples):
        base_prompt = " ".join(data[i]["sentence"].split(" ")[:-1])
        base_query = base_prompt.split(". ")[-1]
        base_query_box_label = base_query.split(" ")[1]
        base_query_box_pos = [
            idx
            for idx, segment in enumerate(base_query.split(". ")[0].split(", "))
            if base_query_box_label in segment
        ][0]
        if base_query_box_pos == -1:
            raise ValueError("Box label not found in the base prompt")
        base_prompts.append(base_prompt)

        source_query_box_pos = base_query_box_pos
        random_choices = list(range(0, num_samples))
        random.shuffle(random_choices)
        while source_query_box_pos == base_query_box_pos:
            random_source_index = random.choice(random_choices)
            source_prompt = " ".join(
                data[random_source_index]["sentence"].split(" ")[:-1]
            )
            source_query = source_prompt.split(". ")[-1]
            source_query_box_label = source_query.split(" ")[1]
            source_query_box_pos = [
                idx
                for idx, segment in enumerate(source_prompt.split(". ")[0].split(", "))
                if source_query_box_label in segment
            ][0]
            random_choices.remove(random_source_index)

        source_prompt = (
            "The apple is in Box O, " + source_prompt[0].lower() + source_prompt[1:]
        )
        source_prompts.append(source_prompt)

        base_prompt = base_prompt.split(". ")[0]
        correct_object = base_prompt.split(", ")[source_query_box_pos].split(" ")[1]
        labels.append(tokenizer.encode(correct_object)[1])

    base_input_tokens = tokenizer(base_prompts, padding=True, return_tensors="pt")[
        "input_ids"
    ]
    base_last_token_indices = (
        tokenizer(base_prompts, padding=True, return_tensors="pt")[
            "attention_mask"
        ].sum(dim=1)
        - 1
    )
    source_input_tokens = tokenizer(source_prompts, padding=True, return_tensors="pt")[
        "input_ids"
    ]
    source_last_token_indices = (
        tokenizer(source_prompts, padding=True, return_tensors="pt")[
            "attention_mask"
        ].sum(dim=1)
        - 1
    )
    output_ids = torch.tensor(labels)

    return (
        base_input_tokens,
        base_last_token_indices,
        source_input_tokens,
        source_last_token_indices,
        output_ids,
    )


def additional_token_btw_box_and_object(
    tokenizer,
    num_samples,
    data_file,
    object_file,
    num_boxes,
    alt_format,
    correct_pred_indices=[],
):
    with open(data_file) as f:
        data = [json.loads(line) for line in f]

    assert num_samples <= len(data)
    base_prompts, source_prompts, labels = [], [], []

    for i in range(0, num_samples):
        base_prompt = " ".join(data[i]["sentence"].split(" ")[:-1])
        base_query = base_prompt.split(". ")[-1]
        base_query_box_label = base_query.split(" ")[1]
        base_query_box_pos = [
            idx
            for idx, segment in enumerate(base_query.split(". ")[0].split(", "))
            if base_query_box_label in segment
        ][0]
        if base_query_box_pos == -1:
            raise ValueError("Box label not found in the base prompt")
        base_prompts.append(base_prompt)

        source_query_box_pos = base_query_box_pos
        random_choices = list(range(0, num_samples))
        random.shuffle(random_choices)
        while source_query_box_pos == base_query_box_pos:
            random_source_index = random.choice(random_choices)
            source_prompt = " ".join(
                data[random_source_index]["sentence"].split(" ")[:-1]
            )
            source_query = source_prompt.split(". ")[-1]
            source_query_box_label = source_query.split(" ")[1]
            source_query_box_pos = [
                idx
                for idx, segment in enumerate(source_prompt.split(". ")[0].split(", "))
                if source_query_box_label in segment
            ][0]
            random_choices.remove(random_source_index)

        source_segment = source_prompt.split(". ")[0].split(", ")[source_query_box_pos]
        source_segment = source_segment.replace(" is in", " is contained in the")
        source_prompt = (
            ", ".join(source_prompt.split(". ")[0].split(", ")[:source_query_box_pos])
            + (", " if source_query_box_pos != 0 else "")
            + source_segment
            + (
                ", "
                if source_query_box_pos
                != len(source_prompt.split(". ")[0].split(", ")) - 1
                else ""
            )
            + ", ".join(
                source_prompt.split(". ")[0].split(", ")[source_query_box_pos + 1 :]
            )
            + ". "
            + source_prompt.split(". ")[-1]
        )
        source_prompts.append(source_prompt)

        base_prompt = base_prompt.split(". ")[0]
        correct_object = base_prompt.split(", ")[source_query_box_pos].split(" ")[1]
        labels.append(tokenizer.encode(correct_object)[1])

    base_input_tokens = tokenizer(base_prompts, padding=True, return_tensors="pt")[
        "input_ids"
    ]
    base_last_token_indices = (
        tokenizer(base_prompts, padding=True, return_tensors="pt")[
            "attention_mask"
        ].sum(dim=1)
        - 1
    )
    source_input_tokens = tokenizer(source_prompts, padding=True, return_tensors="pt")[
        "input_ids"
    ]
    source_last_token_indices = (
        tokenizer(source_prompts, padding=True, return_tensors="pt")[
            "attention_mask"
        ].sum(dim=1)
        - 1
    )
    output_ids = torch.tensor(labels)

    return (
        base_input_tokens,
        base_last_token_indices,
        source_input_tokens,
        source_last_token_indices,
        output_ids,
    )


def diff_index_query_box(
    tokenizer,
    num_samples,
    data_file,
    object_file,
    num_boxes,
    alt_format,
    correct_pred_indices=[],
):
    with open(data_file) as f:
        data = [json.loads(line) for line in f]

    assert num_samples <= len(data)
    base_prompts, source_prompts, labels = [], [], []

    for i in range(0, num_samples):
        base_prompt = " ".join(data[i]["sentence"].split(" ")[:-1])
        base_query = base_prompt.split(". ")[-1]
        base_query_box_label = base_query.split(" ")[1]
        base_query_box_pos = [
            idx
            for idx, segment in enumerate(base_query.split(". ")[0].split(", "))
            if base_query_box_label in segment
        ][0]
        if base_query_box_pos == -1:
            raise ValueError("Box label not found in the base prompt")
        base_prompts.append(base_prompt)

        source_query_box_pos = base_query_box_pos
        random_choices = list(range(0, num_samples))
        random.shuffle(random_choices)
        while source_query_box_pos == base_query_box_pos:
            random_source_index = random.choice(random_choices)
            source_prompt = " ".join(
                data[random_source_index]["sentence"].split(" ")[:-1]
            )
            source_query = source_prompt.split(". ")[-1]
            source_query_box_label = source_query.split(" ")[1]
            source_query_box_pos = [
                idx
                for idx, segment in enumerate(source_prompt.split(". ")[0].split(", "))
                if source_query_box_label in segment
            ][0]
            random_choices.remove(random_source_index)

        source_prompts.append(source_prompt)

        base_prompt = base_prompt.split(". ")[0]
        correct_object = base_prompt.split(", ")[(source_query_box_pos + 1) % 7].split(
            " "
        )[1]
        labels.append(tokenizer.encode(correct_object)[1])

    base_input_tokens = tokenizer(base_prompts, padding=True, return_tensors="pt")[
        "input_ids"
    ]
    base_last_token_indices = (
        tokenizer(base_prompts, padding=True, return_tensors="pt")[
            "attention_mask"
        ].sum(dim=1)
        - 1
    )
    source_input_tokens = tokenizer(source_prompts, padding=True, return_tensors="pt")[
        "input_ids"
    ]
    source_last_token_indices = (
        tokenizer(source_prompts, padding=True, return_tensors="pt")[
            "attention_mask"
        ].sum(dim=1)
        - 1
    )
    output_ids = torch.tensor(labels)

    return (
        base_input_tokens,
        base_last_token_indices,
        source_input_tokens,
        source_last_token_indices,
        output_ids,
    )


def box_object_altered_order(
    tokenizer,
    num_samples,
    data_file,
    object_file,
    num_boxes,
    alt_format,
    correct_pred_indices=[],
):
    with open(data_file) as f:
        data = [json.loads(line) for line in f]

    # data = [data[i] for i in correct_pred_indices]

    assert num_samples <= len(data)
    base_prompts, source_prompts, labels = [], [], []

    for i in range(0, num_samples):
        base_prompt = " ".join(data[i]["sentence"].split(" ")[:-1])
        base_query = base_prompt.split(". ")[-1]
        base_query_box_label = base_query.split(" ")[1]
        base_query_box_pos = [
            idx
            for idx, segment in enumerate(base_query.split(". ")[0].split(", "))
            if base_query_box_label in segment
        ][0]
        if base_query_box_pos == -1:
            raise ValueError("Box label not found in the base prompt")
        base_prompts.append(base_prompt)

        source_query_box_pos = base_query_box_pos
        random_choices = list(range(0, num_samples))
        random.shuffle(random_choices)
        while source_query_box_pos == base_query_box_pos:
            random_source_index = random.choice(random_choices)
            source_prompt = " ".join(
                data[random_source_index]["sentence"].split(" ")[:-1]
            )
            source_query = source_prompt.split(". ")[-1]
            source_query_box_label = source_query.split(" ")[1]
            source_query_box_pos = [
                idx
                for idx, segment in enumerate(source_prompt.split(". ")[0].split(", "))
                if source_query_box_label in segment
            ][0]
            random_choices.remove(random_source_index)

        source_segment = source_prompt.split(". ")[0].split(", ")[source_query_box_pos]
        source_box = source_segment.split(" ")[-1]
        source_object = source_segment.split(" ")[1]
        source_segment = f"Box {source_box} contains the {source_object}"

        source_prompt = (
            ", ".join(source_prompt.split(". ")[0].split(", ")[:source_query_box_pos])
            + ", "
            + source_segment
            + ", "
            + ", ".join(
                source_prompt.split(". ")[0].split(", ")[source_query_box_pos + 1 :]
            )
            + ". "
            + source_prompt.split(". ")[-1]
        )

        source_prompts.append(source_prompt)

        base_prompt = base_prompt.split(". ")[0]
        correct_object = base_prompt.split(", ")[source_query_box_pos].split(" ")[1]
        labels.append(tokenizer.encode(correct_object)[1])

    base_input_tokens = tokenizer(base_prompts, padding=True, return_tensors="pt")[
        "input_ids"
    ]
    base_last_token_indices = (
        tokenizer(base_prompts, padding=True, return_tensors="pt")[
            "attention_mask"
        ].sum(dim=1)
        - 1
    )
    source_input_tokens = tokenizer(source_prompts, padding=True, return_tensors="pt")[
        "input_ids"
    ]
    source_last_token_indices = (
        tokenizer(source_prompts, padding=True, return_tensors="pt")[
            "attention_mask"
        ].sum(dim=1)
        - 1
    )
    output_ids = torch.tensor(labels)

    return (
        base_input_tokens,
        base_last_token_indices,
        source_input_tokens,
        source_last_token_indices,
        output_ids,
    )


def add_comma_after_object(
    tokenizer,
    num_samples,
    data_file,
    object_file,
    num_boxes,
    alt_format,
    correct_pred_indices=[],
):
    with open(data_file) as f:
        data = [json.loads(line) for line in f]

    # data = [data[i] for i in correct_pred_indices]

    assert num_samples <= len(data)
    base_prompts, source_prompts, labels = [], [], []

    for i in range(0, num_samples):
        base_prompt = " ".join(data[i]["sentence"].split(" ")[:-1])
        base_query = base_prompt.split(". ")[-1]
        base_query_box_label = base_query.split(" ")[1]
        base_query_box_pos = [
            idx
            for idx, segment in enumerate(base_query.split(". ")[0].split(", "))
            if base_query_box_label in segment
        ][0]
        if base_query_box_pos == -1:
            raise ValueError("Box label not found in the base prompt")
        base_prompts.append(base_prompt)

        source_query_box_pos = base_query_box_pos
        random_choices = list(range(0, num_samples))
        random.shuffle(random_choices)
        while source_query_box_pos == base_query_box_pos:
            random_source_index = random.choice(random_choices)
            source_prompt = " ".join(
                data[random_source_index]["sentence"].split(" ")[:-1]
            )
            source_query = source_prompt.split(". ")[-1]
            source_query_box_label = source_query.split(" ")[1]
            source_query_box_pos = [
                idx
                for idx, segment in enumerate(source_prompt.split(". ")[0].split(", "))
                if source_query_box_label in segment
            ][0]
            random_choices.remove(random_source_index)

        source_prompt = source_prompt.replace(" is", ", is")
        source_prompts.append(source_prompt)

        base_prompt = base_prompt.split(". ")[0]
        correct_object = base_prompt.split(", ")[source_query_box_pos].split(" ")[1]
        labels.append(tokenizer.encode(correct_object)[1])

    base_input_tokens = tokenizer(base_prompts, padding=True, return_tensors="pt")[
        "input_ids"
    ]
    base_last_token_indices = (
        tokenizer(base_prompts, padding=True, return_tensors="pt")[
            "attention_mask"
        ].sum(dim=1)
        - 1
    )
    source_input_tokens = tokenizer(source_prompts, padding=True, return_tensors="pt")[
        "input_ids"
    ]
    source_last_token_indices = (
        tokenizer(source_prompts, padding=True, return_tensors="pt")[
            "attention_mask"
        ].sum(dim=1)
        - 1
    )
    output_ids = torch.tensor(labels)

    return (
        base_input_tokens,
        base_last_token_indices,
        source_input_tokens,
        source_last_token_indices,
        output_ids,
    )


def remove_comma_desiderata(
    tokenizer,
    num_samples,
    data_file,
    object_file,
    num_boxes,
    alt_format,
    correct_pred_indices=[],
):
    with open(data_file) as f:
        data = [json.loads(line) for line in f]

    # data = [data[i] for i in correct_pred_indices]

    assert num_samples <= len(data)
    base_prompts, source_prompts, labels = [], [], []

    for i in range(0, num_samples):
        base_prompt = " ".join(data[i]["sentence"].split(" ")[:-1])
        base_query = base_prompt.split(". ")[-1]
        base_query_box_label = base_query.split(" ")[1]
        base_query_box_pos = [
            idx
            for idx, segment in enumerate(base_query.split(". ")[0].split(", "))
            if base_query_box_label in segment
        ][0]
        if base_query_box_pos == -1:
            raise ValueError("Box label not found in the base prompt")
        base_prompts.append(base_prompt)

        source_query_box_pos = base_query_box_pos
        random_choices = list(range(0, num_samples))
        random.shuffle(random_choices)
        while source_query_box_pos == base_query_box_pos:
            random_source_index = random.choice(random_choices)
            source_prompt = " ".join(
                data[random_source_index]["sentence"].split(" ")[:-1]
            )
            source_query = source_prompt.split(". ")[-1]
            source_query_box_label = source_query.split(" ")[1]
            source_query_box_pos = [
                idx
                for idx, segment in enumerate(source_prompt.split(". ")[0].split(", "))
                if source_query_box_label in segment
            ][0]
            random_choices.remove(random_source_index)

        source_prompt = source_prompt.replace(", ", " ")
        source_prompts.append(source_prompt)

        base_prompt = base_prompt.split(". ")[0]
        correct_object = base_prompt.split(", ")[source_query_box_pos].split(" ")[1]
        labels.append(tokenizer.encode(correct_object)[1])

    base_input_tokens = tokenizer(base_prompts, padding=True, return_tensors="pt")[
        "input_ids"
    ]
    base_last_token_indices = (
        tokenizer(base_prompts, padding=True, return_tensors="pt")[
            "attention_mask"
        ].sum(dim=1)
        - 1
    )
    source_input_tokens = tokenizer(source_prompts, padding=True, return_tensors="pt")[
        "input_ids"
    ]
    source_last_token_indices = (
        tokenizer(source_prompts, padding=True, return_tensors="pt")[
            "attention_mask"
        ].sum(dim=1)
        - 1
    )
    output_ids = torch.tensor(labels)

    return (
        base_input_tokens,
        base_last_token_indices,
        source_input_tokens,
        source_last_token_indices,
        output_ids,
    )


def box_label_value_desiderata(
    tokenizer,
    num_samples,
    data_file,
    object_file,
    num_boxes,
    alt_format,
    correct_pred_indices=[],
):
    with open(data_file) as f:
        data = [json.loads(line) for line in f]

    assert num_samples <= len(data)
    base_prompts, source_prompts, labels = [], [], []

    for i in range(0, num_samples):
        base_prompt = " ".join(data[i]["sentence"].split(" ")[:-1])
        base_query = base_prompt.split(". ")[-1]
        base_query_box_label = base_query.split(" ")[1]
        base_box_labels = [
            segment.split(" ")[-1] for segment in base_prompt.split(". ")[0].split(", ")
        ]
        base_prompts.append(base_prompt)

        random_choices = list(range(0, num_samples))
        random.shuffle(random_choices)
        while True:
            random_source_index = random.choice(random_choices)
            source_prompt = " ".join(
                data[random_source_index]["sentence"].split(" ")[:-1]
            )
            source_query = source_prompt.split(". ")[-1]
            source_query_box_label = source_query.split(" ")[1]
            if (
                source_query_box_label in base_box_labels
                and source_query_box_label != base_query_box_label
            ):
                break
            random_choices.remove(random_source_index)

        source_prompts.append(source_prompt)
        base_correct_object = [
            segment.split(" ")[1]
            for segment in base_prompt.split(". ")[0].split(", ")
            if source_query_box_label in segment.split(" ")
        ]
        labels.append(tokenizer.encode(base_correct_object)[1])

    base_input_tokens = tokenizer(base_prompts, padding=True, return_tensors="pt")[
        "input_ids"
    ]
    base_last_token_indices = (
        tokenizer(base_prompts, padding=True, return_tensors="pt")[
            "attention_mask"
        ].sum(dim=1)
        - 1
    )
    source_input_tokens = tokenizer(source_prompts, padding=True, return_tensors="pt")[
        "input_ids"
    ]
    source_last_token_indices = (
        tokenizer(source_prompts, padding=True, return_tensors="pt")[
            "attention_mask"
        ].sum(dim=1)
        - 1
    )
    output_ids = torch.tensor(labels)

    return (
        base_input_tokens,
        base_last_token_indices,
        source_input_tokens,
        source_last_token_indices,
        output_ids,
    )


def object_value_desiderata(
    tokenizer,
    num_samples,
    data_file,
    object_file,
    num_boxes,
    alt_format,
    correct_pred_indices=[],
):
    with open(data_file) as f:
        data = [json.loads(line) for line in f]

    # data = [data[i] for i in correct_pred_indices]

    assert num_samples <= len(data)
    base_prompts, source_prompts, labels = [], [], []

    for i in range(0, num_samples):
        base_prompt = " ".join(data[i]["sentence"].split(" ")[:-1])
        base_query = base_prompt.split(". ")[-1]
        base_query_box_label = base_query.split(" ")[1]
        base_query_box_pos = [
            idx
            for idx, segment in enumerate(base_query.split(". ")[0].split(", "))
            if base_query_box_label in segment
        ][0]
        if base_query_box_pos == -1:
            raise ValueError("Box label not found in the base prompt")
        base_prompts.append(base_prompt)

        source_query_box_pos = base_query_box_pos
        random_choices = list(range(0, num_samples))
        random.shuffle(random_choices)
        while source_query_box_pos == base_query_box_pos:
            random_source_index = random.choice(random_choices)
            source_prompt = " ".join(
                data[random_source_index]["sentence"].split(" ")[:-1]
            )
            source_query = source_prompt.split(". ")[-1]
            source_query_box_label = source_query.split(" ")[1]
            source_query_box_pos = [
                idx
                for idx, segment in enumerate(source_prompt.split(". ")[0].split(", "))
                if source_query_box_label in segment
            ][0]
            random_choices.remove(random_source_index)

        source_prompts.append(source_prompt)

        base_prompt = base_prompt.split(". ")[0]
        correct_object = source_prompt.split(", ")[source_query_box_pos].split(" ")[1]
        labels.append(tokenizer.encode(correct_object)[1])

    base_input_tokens = tokenizer(base_prompts, padding=True, return_tensors="pt")[
        "input_ids"
    ]
    base_last_token_indices = (
        tokenizer(base_prompts, padding=True, return_tensors="pt")[
            "attention_mask"
        ].sum(dim=1)
        - 1
    )
    source_input_tokens = tokenizer(source_prompts, padding=True, return_tensors="pt")[
        "input_ids"
    ]
    source_last_token_indices = (
        tokenizer(source_prompts, padding=True, return_tensors="pt")[
            "attention_mask"
        ].sum(dim=1)
        - 1
    )
    output_ids = torch.tensor(labels)

    return (
        base_input_tokens,
        base_last_token_indices,
        source_input_tokens,
        source_last_token_indices,
        output_ids,
    )


def positional_desiderata(
    tokenizer,
    num_samples,
    data_file,
    object_file,
    num_boxes,
    alt_format,
    correct_pred_indices=[],
):
    with open(data_file) as f:
        data = [json.loads(line) for line in f]

    # data = [data[i] for i in correct_pred_indices]

    assert num_samples <= len(data)
    base_prompts, source_prompts, labels = [], [], []

    for i in range(0, num_samples):
        base_prompt = " ".join(data[i]["sentence"].split(" ")[:-1])
        base_query = base_prompt.split(". ")[-1]
        base_query_box_label = base_query.split(" ")[1]
        base_query_box_pos = [
            idx
            for idx, segment in enumerate(base_query.split(". ")[0].split(", "))
            if base_query_box_label in segment
        ][0]
        if base_query_box_pos == -1:
            raise ValueError("Box label not found in the base prompt")
        base_prompts.append(base_prompt)

        source_query_box_pos = base_query_box_pos
        random_choices = list(range(0, num_samples))
        random.shuffle(random_choices)
        while source_query_box_pos == base_query_box_pos:
            random_source_index = random.choice(random_choices)
            source_prompt = " ".join(
                data[random_source_index]["sentence"].split(" ")[:-1]
            )
            source_query = source_prompt.split(". ")[-1]
            source_query_box_label = source_query.split(" ")[1]
            source_query_box_pos = [
                idx
                for idx, segment in enumerate(source_prompt.split(". ")[0].split(", "))
                if source_query_box_label in segment
            ][0]
            random_choices.remove(random_source_index)

        source_prompts.append(source_prompt)

        base_prompt = base_prompt.split(". ")[0]
        correct_object = base_prompt.split(", ")[source_query_box_pos].split(" ")[1]
        labels.append(tokenizer.encode(correct_object)[1])

    base_input_tokens = tokenizer(base_prompts, padding=True, return_tensors="pt")[
        "input_ids"
    ]
    base_last_token_indices = (
        tokenizer(base_prompts, padding=True, return_tensors="pt")[
            "attention_mask"
        ].sum(dim=1)
        - 1
    )
    source_input_tokens = tokenizer(source_prompts, padding=True, return_tensors="pt")[
        "input_ids"
    ]
    source_last_token_indices = (
        tokenizer(source_prompts, padding=True, return_tensors="pt")[
            "attention_mask"
        ].sum(dim=1)
        - 1
    )
    output_ids = torch.tensor(labels)

    return (
        base_input_tokens,
        base_last_token_indices,
        source_input_tokens,
        source_last_token_indices,
        output_ids,
    )


def sample_box_data(tokenizer, num_samples, data_file):
    """
    Sample data from the box data file

    Args:
        tokenizer: Tokenizer to be used
        num_samples: Number of samples to be generated
        data_file: Path to the box data file
    """

    with open(data_file, encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    assert num_samples <= len(data)
    prompts, labels = [], []

    for i in range(num_samples):
        label = data[i]["sentence"].split(" ")[-1][:-1]
        prompt = " ".join(data[i]["sentence"].split(" ")[:-1])
        prompts.append(prompt)

        labels.append(tokenizer.encode(label)[1])

    input_tokens = tokenizer(prompts, padding=True, return_tensors="pt")
    last_token_indices = input_tokens["attention_mask"].sum(dim=1) - 1
    output_ids = torch.tensor(labels)
    input_ids = input_tokens["input_ids"]

    return input_ids, last_token_indices, output_ids


def load_pp_data(
    model,
    tokenizer,
    num_samples,
    data_file,
    num_boxes,
):
    """
    Load data for path patching task consisting of original and counterfactual
    examples (random label and random object).
    """
    (
        input_ids,
        last_token_indices,
        output_ids,
    ) = sample_box_data(tokenizer, num_samples, data_file)

    all_base_input_ids = []
    all_base_input_last_pos = []
    all_source_input_ids = []
    all_source_input_last_pos = []
    all_ctf_output_ids = []
    all_intervention_ids = []
    all_incorrect_output_ids = []

    for i in range(0, num_samples, num_boxes):
        for j in range(num_boxes):
            if i + j >= num_samples:
                break

            all_base_input_ids += [input_ids[i + j]]
            all_base_input_last_pos += [last_token_indices[i + j]]
            all_ctf_output_ids += [output_ids[i + j]]

            random_source_index = random.choice(
                list(range(0, num_samples - ((j + 1) % num_boxes), num_boxes))
            )
            random_source_index += (j + 1) % num_boxes
            source_example = input_ids[random_source_index].clone()

            # Change the query box label with a random alphabet
            random_alphabet = chr(random.randint(65, 90))
            random_alphabet_token = tokenizer(
                random_alphabet, return_tensors="pt"
            ).input_ids[0, 1]
            source_example[-3] = random_alphabet_token

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
