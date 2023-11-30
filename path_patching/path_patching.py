import torch
import transformers
import fire
from functools import partial
from baukit import TraceDict
from tqdm import tqdm

from pp_utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int):
    """
    Sets the seed for reproducibility.

    Args:
        seed (int): Seed to use.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    transformers.set_seed(seed)


def apply_pp(
    model: AutoModelForCausalLM = None,
    clean_cache: dict = None,
    corrupt_cache: dict = None,
    dataloader: DataLoader = None,
    receiver_heads: list = None,
    receiver_layers: list = None,
    clean_logit_outputs: dict = None,
    hook_points: list = None,
    rel_pos: int = None,
):
    """
    Applies Path Patching from all heads to receiver heads and returns the patching scores for each head.

    Args:
        model (AutoModelForCausalLM): Model to apply Path Patching to.
        clean_cache (dict): Clean cache for the model.
        corrupt_cache (dict): Corrupt cache for the model.
        dataloader (DataLoader): Dataloader for the model.
        receiver_heads (list): List of receiver heads.
        receiver_layers (list): List of receiver layers.
        clean_logit_outputs (dict): Clean logit outputs for the model.
        hook_points (list): List of hook points.
        rel_pos (int): Relative position of the receiver heads.
    """

    path_patching_score = torch.zeros(
        model.config.num_hidden_layers, model.config.num_attention_heads
    ).to(device)
    apply_softmax = torch.nn.Softmax(dim=-1)

    for layer in tqdm(range(model.config.num_hidden_layers)):
        for head in range(model.config.num_attention_heads):
            with torch.no_grad():
                for bi, inp in enumerate(dataloader):
                    batch_size = inp["base_tokens"].shape[0]

                    for k, v in inp.items():
                        if v is not None and isinstance(v, torch.Tensor):
                            inp[k] = v.to(model.device)

                    # Step 2: Patch the output of the sender head and store the input of the receiver heads
                    with TraceDict(
                        model,
                        hook_points + receiver_layers,
                        retain_input=True,
                        edit_output=partial(
                            patching_heads,
                            model=model,
                            clean_cache=clean_cache[bi],
                            corrupt_cache=corrupt_cache[bi],
                            base_tokens=inp["base_tokens"],
                            sender_layer=layer,
                            sender_head=head,
                            clean_last_token_indices=inp["base_last_token_indices"],
                            corrupt_last_token_indices=inp["source_last_token_indices"],
                            rel_pos=rel_pos,
                            batch_size=batch_size,
                        ),
                    ) as patched_cache:
                        patched_out = model(inp["base_tokens"])

                    # Step 3: Patch the input of the receiver head and compute the patching score
                    if len(receiver_layers) != 0:
                        with TraceDict(
                            model,
                            receiver_layers,
                            retain_input=False,
                            edit_output=partial(
                                patching_receiver_heads,
                                model=model,
                                base_tokens=inp["base_tokens"],
                                patched_cache=patched_cache,
                                receiver_heads=receiver_heads,
                                clean_last_token_indices=inp["base_last_token_indices"],
                                rel_pos=rel_pos,
                                batch_size=batch_size,
                            ),
                        ) as _:
                            patched_out = model(inp["base_tokens"])

                    # Compute the patching score
                    for i in range(batch_size):
                        logits = apply_softmax(
                            patched_out.logits[i, inp["base_last_token_indices"][i]]
                        )
                        # patching_score = (p_patch - p_clean) / p_clean
                        score = (
                            (logits[inp["labels"][i]]).item()
                            - (clean_logit_outputs[bi][i])
                        ) / clean_logit_outputs[bi][i]

                        path_patching_score[layer, head] = score

                    del patched_out, patched_cache, inp
                    torch.cuda.empty_cache()

    return path_patching_score


def pp_main(
    datafile: str = "../box_datasets/no_instructions/alternative/Random/7/train.jsonl",
    num_boxes: int = 7,
    model_name: str = "llama",
    num_samples: int = 500,
    n_value_fetcher: int = 50,
    n_pos_trans: int = 15,
    n_pos_detect: int = 30,
    n_struct_read: int = 5,
    output_path: str = "./results/",
    seed: int = 10,
    batch_size: int = 100,
):
    """
    Main function to run Path Patching.

    Args:

        datafile (str): Path to the dataset.
        num_boxes (int): Number of boxes in the dataset.
        model_name (str): Name of the model to use.
        num_samples (int): Number of samples to use from the dataset.
        n_value_fetcher (int): Number of Value Fetcher heads to select.
        n_pos_trans (int): Number of Position Transformer heads to select.
        n_pos_detect (int): Number of Position Detector heads to select.
        n_struct_read (int): Number of Structural Reader heads to select.
        output_path (str): Path to store the results.
        seed (int): Seed to use.
        batch_size (int): Batch size to use.
    """

    set_seed(seed)

    model, tokenizer = get_model_and_tokenizer(model_name)
    print("MODEL AND TOKENIZER LOADED")

    dataloader = load_dataloader(
        tokenizer=tokenizer,
        datafile=datafile,
        num_samples=num_samples,
        num_boxes=num_boxes,
        batch_size=batch_size,
    )
    print("DATALOADER CREATED")

    # Step 1: Compute clean and corrupt caches
    (
        clean_cache,
        corrupt_cache,
        clean_logit_outputs,
        _,
        hook_points,
    ) = get_caches(model, dataloader)

    # Compute Value Fetcher Heads
    print("COMPUTING VALUE FETCHER HEADS...")
    patching_scores = apply_pp(
        model=model,
        clean_cache=clean_cache,
        corrupt_cache=corrupt_cache,
        dataloader=dataloader,
        receiver_heads=[],
        receiver_layers=[],
        clean_logit_outputs=clean_logit_outputs,
        hook_points=hook_points,
        rel_pos=0,
    )
    torch.save(patching_scores, output_path + "direct_logit_heads.pt")
    value_fetcher_heads = compute_topk_components(
        patching_scores=patching_scores, k=n_value_fetcher, largest=False
    )
    print(f"VALUE FETCHER HEADS: {value_fetcher_heads}\n")

    # Compute Position Transformer Heads
    print("COMPUTING POSITION TRANSMITTER HEADS...")
    receiver_layers = get_receiver_layers(
        model=model, receiver_heads=value_fetcher_heads, composition="q"
    )
    patching_scores = apply_pp(
        model=model,
        clean_cache=clean_cache,
        corrupt_cache=corrupt_cache,
        dataloader=dataloader,
        receiver_heads=value_fetcher_heads,
        receiver_layers=receiver_layers,
        clean_logit_outputs=clean_logit_outputs,
        hook_points=hook_points,
        rel_pos=0,
    )
    torch.save(patching_scores, output_path + "heads_affect_direct_logit.pt")
    heads_affect_direct_logit = compute_topk_components(
        patching_scores=patching_scores, k=n_pos_trans, largest=False
    )
    print(f"POSITION TRANSMITTER HEADS: {heads_affect_direct_logit}\n")

    # Compute Position Detector Heads
    print("COMPUTING POSITION DETECTOR HEADS...")
    receiver_layers = get_receiver_layers(
        model=model, receiver_heads=heads_affect_direct_logit, composition="v"
    )
    patching_scores = apply_pp(
        model=model,
        clean_cache=clean_cache,
        corrupt_cache=corrupt_cache,
        dataloader=dataloader,
        receiver_heads=heads_affect_direct_logit,
        receiver_layers=receiver_layers,
        clean_logit_outputs=clean_logit_outputs,
        hook_points=hook_points,
        rel_pos=2,
    )
    torch.save(patching_scores, output_path + "heads_at_query_box_pos.pt")
    head_at_query_box_token = compute_topk_components(
        patching_scores=patching_scores, k=n_pos_detect, largest=False
    )
    print(f"POSITION DETECTOR HEADS: {head_at_query_box_token}\n")

    # Compute Structural Reader Heads
    print("COMPUTING STRUCTURAL READER HEADS...")
    receiver_layers = get_receiver_layers(
        model=model, receiver_heads=head_at_query_box_token, composition="v"
    )
    patching_scores = apply_pp(
        model=model,
        clean_cache=clean_cache,
        corrupt_cache=corrupt_cache,
        dataloader=dataloader,
        receiver_heads=head_at_query_box_token,
        receiver_layers=receiver_layers,
        clean_logit_outputs=clean_logit_outputs,
        hook_points=hook_points,
        rel_pos=-1,
    )
    torch.save(patching_scores, output_path + "heads_at_prev_query_box_pos.pt")
    heads_at_prev_box_pos = compute_topk_components(
        patching_scores=patching_scores, k=n_struct_read, largest=False
    )
    print(f"STRUCTURAL READER HEADS: {heads_at_prev_box_pos}\n")


if __name__ == "__main__":
    fire.Fire(pp_main)
