import torch
import transformers
import fire
from functools import partial
from baukit import TraceDict
from tqdm import tqdm

from pp_utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    transformers.set_seed(seed)


def apply_pp(
    model: AutoModelForCausalLM = None,
    clean_cache: dict = None,
    corrupt_cache: dict = None,
    base_tokens: list = None,
    base_last_token_indices: list = None,
    source_last_token_indices: list = None,
    correct_answer_token: list = None,
    receiver_heads: list = None,
    receiver_layers: list = None,
    hook_points: list = None,
    rel_pos: int = None,
):
    path_patching_logits = torch.zeros(
        model.config.num_hidden_layers, model.config.num_attention_heads
    ).to(device)
    batch_size = base_tokens.size(0)
    apply_softmax = torch.nn.Softmax(dim=-1)

    for layer in tqdm(range(model.config.num_hidden_layers)):
        for head in range(model.config.num_attention_heads):
            with torch.no_grad():
                # Step 2
                with TraceDict(
                    model,
                    hook_points + receiver_layers,
                    retain_input=True,
                    edit_output=partial(
                        patching_heads,
                        model=model,
                        clean_cache=clean_cache,
                        corrupt_cache=corrupt_cache,
                        base_tokens=base_tokens,
                        sender_layer=layer,
                        sender_head=head,
                        clean_last_token_indices=base_last_token_indices,
                        corrupt_last_token_indices=source_last_token_indices,
                        rel_pos=rel_pos,
                    ),
                ) as patched_cache:
                    patched_out = model(base_tokens)

                if len(receiver_layers) != 0:
                    # Step 3
                    with TraceDict(
                        model,
                        receiver_layers,
                        retain_input=True,
                        edit_output=partial(
                            patching_receiver_heads,
                            model=model,
                            base_tokens=base_tokens,
                            patched_cache=patched_cache,
                            receiver_heads=receiver_heads,
                            clean_last_token_indices=base_last_token_indices,
                            rel_pos=rel_pos,
                        ),
                    ) as _:
                        patched_out = model(base_tokens)

                for bi in range(batch_size):
                    logits = apply_softmax(
                        patched_out.logits[bi, base_last_token_indices[bi]]
                    )
                    path_patching_logits[layer, head] += (
                        logits[correct_answer_token[bi]]
                    ).item()

                path_patching_logits[layer, head] = (
                    path_patching_logits[layer, head] / batch_size
                )

    del patched_out
    torch.cuda.empty_cache()

    return path_patching_logits


def pp_main(
    datafile: str = "../box_datasets/no_instructions/alternative/Random/7/train.jsonl",
    num_boxes: int = 7,
    model_name: str = "llama",
    num_samples: int = 100,
    n_value_fetcher: int = 20,
    n_pos_trans: int = 10,
    n_pos_detect: int = 10,
    n_struct_read: int = 5,
    output_path: str = f"./results/",
    seed: int = 10,
):
    set_seed(seed)

    model, tokenizer = load_model_tokenizer(model_name)
    (
        base_tokens,
        base_last_token_indices,
        source_tokens,
        source_last_token_indices,
        correct_answer_token,
    ) = load_pp_data(
        tokenizer=tokenizer,
        datafile=datafile,
        num_samples=num_samples,
        num_boxes=num_boxes,
    )

    base_tokens = torch.cat([t.unsqueeze(dim=0) for t in base_tokens], dim=0).to(device)
    source_tokens = torch.cat([t.unsqueeze(dim=0) for t in source_tokens], dim=0).to(
        device
    )
    clean_cache, corrupt_cache, hook_points = get_caches(
        model, base_tokens, source_tokens
    )

    # Compute Value Fetcher Heads
    print("Computing Value Fetcher Heads...")
    patching_scores = apply_pp(
        model=model,
        clean_cache=clean_cache,
        corrupt_cache=corrupt_cache,
        base_tokens=base_tokens,
        base_last_token_indices=base_last_token_indices,
        source_last_token_indices=source_last_token_indices,
        correct_answer_token=correct_answer_token,
        receiver_heads=[],
        receiver_layers=[],
        hook_points=hook_points,
        rel_pos=0,
    )
    torch.save(patching_scores, output_path + "direct_logit_heads.pt")
    direct_logit_heads = compute_topk_components(
        patching_scores=patching_scores, k=n_value_fetcher, largest=False
    )
    print(f"Direct Logit Heads: {direct_logit_heads}\n")

    # Compute Position Transformer Heads
    print("Computing Position Transformer Heads...")
    receiver_layers = get_receiver_layers(
        model=model, receiver_heads=direct_logit_heads
    )
    patching_scores = apply_pp(
        model=model,
        clean_cache=clean_cache,
        corrupt_cache=corrupt_cache,
        base_tokens=base_tokens,
        base_last_token_indices=base_last_token_indices,
        source_last_token_indices=source_last_token_indices,
        correct_answer_token=correct_answer_token,
        receiver_heads=direct_logit_heads,
        receiver_layers=receiver_layers,
        hook_points=hook_points,
        rel_pos=0,
    )
    torch.save(patching_scores, output_path + "heads_affect_direct_logit.pt")
    heads_affect_direct_logit = compute_topk_components(
        patching_scores=patching_scores, k=n_pos_trans, largest=False
    )
    print(f"Heads Affecting Direct Logit: {heads_affect_direct_logit}\n")

    # Compute Position Detector Heads
    print("Computing Position Detector Heads...")
    receiver_layers = get_receiver_layers(
        model=model, receiver_heads=heads_affect_direct_logit
    )
    patching_scores = apply_pp(
        model=model,
        clean_cache=clean_cache,
        corrupt_cache=corrupt_cache,
        base_tokens=base_tokens,
        base_last_token_indices=base_last_token_indices,
        source_last_token_indices=source_last_token_indices,
        correct_answer_token=correct_answer_token,
        receiver_heads=heads_affect_direct_logit,
        receiver_layers=receiver_layers,
        hook_points=hook_points,
        rel_pos=2,
    )
    torch.save(patching_scores, output_path + "heads_at_query_box_pos.pt")
    head_at_query_box_token = compute_topk_components(
        patching_scores=patching_scores, k=n_pos_detect, largest=False
    )
    print(f"Heads at Query Box Position: {head_at_query_box_token}\n")

    # Compute Structural Reader Heads
    print("Computing Structural Reader Heads...")
    receiver_layers = get_receiver_layers(
        model=model, receiver_heads=head_at_query_box_token
    )
    patching_scores = apply_pp(
        model=model,
        clean_cache=clean_cache,
        corrupt_cache=corrupt_cache,
        base_tokens=base_tokens,
        base_last_token_indices=base_last_token_indices,
        source_last_token_indices=source_last_token_indices,
        correct_answer_token=correct_answer_token,
        receiver_heads=head_at_query_box_token,
        receiver_layers=receiver_layers,
        hook_points=hook_points,
        rel_pos=-1,
    )
    torch.save(patching_scores, output_path + "heads_at_prev_query_box_pos.pt")
    heads_at_prev_box_pos = compute_topk_components(
        patching_scores=patching_scores, k=n_struct_read, largest=False
    )
    print(f"Heads at Previous Box Position: {heads_at_prev_box_pos}\n")


if __name__ == "__main__":
    fire.Fire(pp_main)
