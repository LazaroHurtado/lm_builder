import math
import torch
import torch.nn as nn
import gc

from lm_builder import LanguageModel
from lm_builder.transformer import TransformerConfig
from lm_builder.utils import change_state_dict_names

from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "cpu"
MODEL_ARCH_FILE = "examples/gpt2_xl.yml"
HF_MODEL_NAME = "openai-community/gpt2-xl"


class NewGELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

def load_from_state_dict(original_state_dict):
    # Naming conventions between gpt2 and this framework differ
    name_changes = [
        (".h.", ".blocks."),
        ("mlp.c_fc", "ffn.up_proj"),
        ("mlp.c_proj", "ffn.down_proj"),
        ("ln_1.", "attn_norm."),
        ("ln_2.", "ffn_norm."),
        ("ln_f.", "norm."),
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

    to_partition = ".c_attn."

    return change_state_dict_names(
        original_state_dict, name_changes, to_transpose, to_partition
    )


def main():
    # Load gpt2-xl model from huggingface to get the state dict
    model_hf = AutoModelForCausalLM.from_pretrained(HF_MODEL_NAME)
    state_dict = model_hf.state_dict()

    new_state_dict = load_from_state_dict(state_dict)
    del state_dict, model_hf
    gc.collect()

    transformer_config = TransformerConfig.from_yml(MODEL_ARCH_FILE)
    transformer_config.ffn.activation_fn = NewGELU()

    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)

    with torch.no_grad():
        gpt2_xl = LanguageModel(transformer_config, tokenizer, device=DEVICE)
        gpt2_xl.load_state_dict(new_state_dict)

    del new_state_dict
    gc.collect()

    gpt2_xl.to(DEVICE)
    gpt2_xl.eval()

    gpt2_xl.prompt("Claude Shannon, the", top_k=2, max_new_tokens=20, temperature=0.9)


if __name__ == "__main__":
    main()
