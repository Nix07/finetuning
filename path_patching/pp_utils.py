import torch
import sys
import transformers
from transformers import (
    AutoModelForCausalLM,
    LlamaForCausalLM,
    LlamaTokenizer,
)
from baukit import TraceDict
from einops import rearrange, einsum
from peft import PeftModel

sys.path.append("/data/nikhil_prakash/anima-2.0/")
import analysis_utils
from counterfactual_datasets.entity_tracking import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed = 10
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
transformers.set_seed(seed)


def compute_topk_components(patching_scores: torch.Tensor, k: int, largest=True):
    """Computes the topk most influential components (i.e. heads) for patching."""
    top_indices = torch.topk(patching_scores.flatten(), k, largest=largest).indices

    # Convert the top_indices to 2D indices
    row_indices = top_indices // patching_scores.shape[1]
    col_indices = top_indices % patching_scores.shape[1]
    top_components = torch.stack((row_indices, col_indices), dim=1)
    # Get the top indices as a list of 2D indices (row, column)
    top_components = top_components.tolist()
    return top_components


def load_model_tokenizer(model_name: str):
    print(f"Loading model...")
    tokenizer = LlamaTokenizer.from_pretrained(
        "hf-internal-testing/llama-tokenizer", padding_side="right"
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    if model_name == "llama":
        path = "../llama_7b/"
        model = AutoModelForCausalLM.from_pretrained(path).to(device)

    elif model_name == "goat":
        base_model = "decapoda-research/llama-7b-hf"
        lora_weights = "tiedong/goat-lora-7b"
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=False,
            torch_dtype=torch.float32,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float32,
            device_map={"": 0},
        )

    elif model_name == "vicuna":
        path = "AlekseyKorshuk/vicuna-7b"
        model = AutoModelForCausalLM.from_pretrained(path).to(device)

    return model, tokenizer


def load_pp_data(
    tokenizer: LlamaTokenizer, datafile: str, num_samples: int, num_boxes: int
):
    print(f"Loading dataset...")
    raw_data = box_index_aligner_examples(
        tokenizer,
        num_samples=num_samples,
        data_file=datafile,
        architecture="LLaMAForCausalLM",
        few_shot=False,
        alt_examples=True,
        num_ents_or_ops=num_boxes,
    )
    base_tokens = raw_data[0]
    base_last_token_indices = raw_data[1]
    source_tokens = raw_data[2]
    source_last_token_indices = raw_data[3]
    correct_answer_token = raw_data[4]

    return (
        base_tokens,
        base_last_token_indices,
        source_tokens,
        source_last_token_indices,
        correct_answer_token,
    )


def get_caches(model: AutoModelForCausalLM, base_tokens: list, source_tokens: list):
    if model.config.architectures[0] == "LlamaForCausalLM":
        hook_points = [
            f"model.layers.{layer}.self_attn.o_proj"
            for layer in range(model.config.num_hidden_layers)
        ]
    else:
        hook_points = [
            f"base_model.model.model.layers.{layer}.self_attn.o_proj"
            for layer in range(model.config.num_hidden_layers)
        ]

    with torch.no_grad():
        with TraceDict(
            model,
            hook_points,
            retain_input=True,
        ) as clean_cache:
            _ = model(base_tokens)

        with TraceDict(
            model,
            hook_points,
            retain_input=True,
        ) as corrupt_cache:
            _ = model(source_tokens)

    return clean_cache, corrupt_cache, hook_points


def patching_heads(
    inputs=None,
    output=None,
    layer: str = None,
    model: AutoModelForCausalLM = None,
    clean_cache: dict = None,
    corrupt_cache: dict = None,
    base_tokens: list = None,
    sender_layer: str = None,
    sender_head: str = None,
    clean_last_token_indices: list = None,
    corrupt_last_token_indices: list = None,
    rel_pos: int = None,
):
    """
    rel_pos: Represents the token position relative to the "real" (non-padded) last token in the sequence. All the heads at this position and subsequent positions need to patched from clean run, except the sender head at this position.
    """

    input = inputs[0]
    batch_size = input.size(0)

    if "o_proj" in layer:
        input = rearrange(
            input,
            "batch seq_len (n_heads d_head) -> batch seq_len n_heads d_head",
            n_heads=model.config.num_attention_heads,
        )
        clean_head_outputs = rearrange(
            clean_cache[layer].input,
            "batch seq_len (n_heads d_head) -> batch seq_len n_heads d_head",
            n_heads=model.config.num_attention_heads,
        )
        corrupt_head_outputs = rearrange(
            corrupt_cache[layer].input,
            "batch seq_len (n_heads d_head) -> batch seq_len n_heads d_head",
            n_heads=model.config.num_attention_heads,
        )

        if model.config.architectures[0] == "LlamaForCausalLM":
            layer_idx = int(layer.split(".")[2])
        else:
            layer_idx = int(layer.split(".")[4])
        if sender_layer == layer_idx:
            for bi in range(batch_size):
                if rel_pos == -1:
                    # Computing the previous query box label token position
                    clean_prev_box_label_pos = (
                        analysis_utils.compute_prev_query_box_pos(
                            base_tokens[bi], clean_last_token_indices[bi]
                        )
                    )

                    # Since, queery box is not present in the prompt, patch in
                    # the output of heads from any random box label token, i.e. `clean_prev_box_label_pos`
                    corrupt_prev_box_label_pos = clean_prev_box_label_pos
                else:
                    clean_prev_box_label_pos = clean_last_token_indices[bi] - rel_pos
                    corrupt_prev_box_label_pos = (
                        corrupt_last_token_indices[bi] - rel_pos
                    )

                for pos in range(
                    clean_prev_box_label_pos, clean_last_token_indices[bi] + 1
                ):
                    for head_ind in range(model.config.num_attention_heads):
                        if head_ind == sender_head and pos == clean_prev_box_label_pos:
                            input[bi, pos, sender_head] = corrupt_head_outputs[
                                bi, corrupt_prev_box_label_pos, sender_head
                            ]
                        else:
                            input[bi, pos, head_ind] = clean_head_outputs[
                                bi, pos, head_ind
                            ]

        else:
            for bi in range(batch_size):
                if rel_pos == -1:
                    # Computing the previous query box label token position
                    clean_prev_box_label_pos = (
                        analysis_utils.compute_prev_query_box_pos(
                            base_tokens[bi], clean_last_token_indices[bi]
                        )
                    )
                else:
                    clean_prev_box_label_pos = clean_last_token_indices[bi] - rel_pos

                for pos in range(
                    clean_prev_box_label_pos, clean_last_token_indices[bi] + 1
                ):
                    input[bi, pos] = clean_head_outputs[bi, pos]

        input = rearrange(
            input,
            "batch seq_len n_heads d_head -> batch seq_len (n_heads d_head)",
            n_heads=model.config.num_attention_heads,
        )

        w_o = model.state_dict()[f"{layer}.weight"]
        output = einsum(
            input,
            w_o,
            "batch seq_len hidden_size, d_model hidden_size -> batch seq_len d_model",
        )

    return output


def patching_receiver_heads(
    output=None,
    layer=None,
    model: AutoModelForCausalLM = None,
    base_tokens: list = None,
    patched_cache: dict = None,
    receiver_heads: list = None,
    clean_last_token_indices: list = None,
    rel_pos: int = None,
):
    batch_size = output.size(0)
    if model.config.architectures[0] == "LlamaForCausalLM":
        receiver_heads_in_curr_layer = [
            h for l, h in receiver_heads if l == int(layer.split(".")[2])
        ]
    else:
        receiver_heads_in_curr_layer = [
            h for l, h in receiver_heads if l == int(layer.split(".")[4])
        ]

    output = rearrange(
        output,
        "batch seq_len (n_heads d_head) -> batch seq_len n_heads d_head",
        n_heads=model.config.num_attention_heads,
    )
    patched_head_outputs = rearrange(
        patched_cache[layer].output,
        "batch seq_len (n_heads d_head) -> batch seq_len n_heads d_head",
        n_heads=model.config.num_attention_heads,
    )

    # Patch in the output of the receiver heads from patched run
    for receiver_head in receiver_heads_in_curr_layer:
        for bi in range(batch_size):
            if rel_pos == -1:
                # Computing the previous query box label token position
                clean_prev_box_label_pos = analysis_utils.compute_prev_query_box_pos(
                    base_tokens[bi], clean_last_token_indices[bi]
                )
            else:
                clean_prev_box_label_pos = clean_last_token_indices[bi] - rel_pos

            output[bi, clean_prev_box_label_pos, receiver_head] = patched_head_outputs[
                bi, clean_prev_box_label_pos, receiver_head
            ]

    output = rearrange(
        output,
        "batch seq_len n_heads d_head -> batch seq_len (n_heads d_head)",
        n_heads=model.config.num_attention_heads,
    )

    return output


def get_receiver_layers(receiver_heads):
    receiver_layers = list(
        set([f"model.layers.{layer}.self_attn.v_proj" for layer, _ in receiver_heads])
    )
    return receiver_layers
