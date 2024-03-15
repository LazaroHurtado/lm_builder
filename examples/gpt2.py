import tiktoken
import gc

from lm_builder import LanguageModel
from lm_builder.transformer import TransformerConfig
from lm_builder.utils import change_state_dict_names

from transformers import GPT2LMHeadModel

# Consider changing this to "cuda" for Nvidia GPUs or "mps" for Apple Metal GPUs
DEVICE = "cpu"
MODEL_ARCH_FILE = "examples/gpt2_xl.yml"
HF_MODEL_NAME = "gpt2-xl"

CONFIG = TransformerConfig.from_yml(MODEL_ARCH_FILE)


def load_from_state_dict(original_state_dict):
    # Naming conventions between gpt2 and this framework differ
    name_changes = [
        (".h.", ".blocks."),
        ("mlp.c_fc", "ffn.up_proj"),
        ("mlp.c_proj", "ffn.down_proj"),
        ("ln_1.", "attn_norm."),
        ("ln_2.", "ffn_norm."),
        ("ln_f.", "norm."),
        ("wpe.weight", "wpe._pos_embs_cached"),
        ("attn.c_attn", "attn.qkv_proj"),
        ("attn.c_proj", "attn.out_proj"),
    ]
    # GPT2 used convolutions instead of linear modules so these weights won't match
    # our linear layer weights, instead we have to transpose them.
    to_transpose = [
        "attn.c_attn.weight",
        "attn.c_proj.weight",
        "mlp.c_fc.weight",
        "mlp.c_proj.weight",
    ]

    # The language model head is used to convert from embedding dimension to vocab size, notice
    # how this is the same as the token embedding linear layer but backwards so we can just use
    # the same weights but transposed incase our model doesnt come with it.
    if "lm_head.weight" not in original_state_dict:
        original_state_dict["lm_head.weight"] = original_state_dict[
            "transformer.wte.weight"
        ]

    return change_state_dict_names(original_state_dict, name_changes, to_transpose)


def main():
    # Load gpt2-xl model from huggingface to get the state dict
    model_hf = GPT2LMHeadModel.from_pretrained(HF_MODEL_NAME)
    state_dict = model_hf.state_dict()

    # Get a state dict with our naming conventions
    new_state_dict = load_from_state_dict(state_dict)
    del model_hf
    del state_dict
    gc.collect()

    tokenizer = tiktoken.encoding_for_model("gpt2")
    gpt2_xl = LanguageModel(CONFIG, tokenizer, device=DEVICE)

    gpt2_xl.load_state_dict(gpt2_xl.state_dict() | new_state_dict)
    del new_state_dict
    gc.collect()

    gpt2_xl.to(DEVICE)
    gpt2_xl.eval()

    gpt2_xl.prompt("Claude Shannon, the", top_k=2, max_new_tokens=20, temperature=0.9)


if __name__ == "__main__":
    main()
