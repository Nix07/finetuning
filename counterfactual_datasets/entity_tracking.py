import json
import torch


def entity_tracking_example_sampler(tokenizer, num_samples, num_ops=3):
    with open(f"./box_datasets/no_instructions/{num_ops}/train.jsonl") as f:
        data = [json.loads(line) for line in f]

    assert num_samples <= len(data)
    prompts, labels = [], []

    for i in range(num_samples):
        prompts.append(" ".join(data[i]["sentence"].split(" ")[:-1]))
        label = data[i]["sentence"].split(" ")[-1][:-1]
        # 0th index will be BOS token for llama-like tokenizer
        labels.append(tokenizer.encode(label)[1])

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


def box_name_alignment_example_sampler(tokenizer, num_samples, num_ops=3):
    input_ids, last_token_indices, output_ids = entity_tracking_example_sampler(
        tokenizer, num_samples, num_ops
    )

    all_base_input_ids = []
    all_base_input_last_pos = []
    all_source_input_ids = []
    all_source_input_last_pos = []
    all_ctf_output_ids = []
    all_intervention_ids = []

    for i in range(0, len(input_ids), num_ops):
        for j in range(num_ops):
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


def name_alignment_sampler(
    tokenizer, max_n_training_examples, bound_functors, num_ops=3
):
    (
        all_base_input_ids,
        all_base_input_last_pos,
        all_source_input_ids,
        all_source_input_last_pos,
        all_ctf_output_ids,
        all_intervention_ids,
    ) = bound_functors(
        tokenizer,
        max_n_training_examples,
        num_ops,
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
    game="entity_tracking",
):
    all_input_ids = []
    all_last_token_indices = []
    all_output_ids = []  # this one does not have input ids, etc..

    if game == "entity_tracking":
        (
            all_input_ids,
            all_last_token_indices,
            all_output_ids,
        ) = entity_tracking_example_sampler(
            tokenizer, max_n_training_examples, num_ops=3
        )

    return all_input_ids, all_last_token_indices, all_output_ids
