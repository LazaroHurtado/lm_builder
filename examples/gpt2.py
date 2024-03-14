import math
import time
import tiktoken
import torch
import gc

from lm_builder.ffn import FeedForwardConfig
from lm_builder.transformer import Transformer, TransformerConfig
from lm_builder.utils import change_state_dict_names

from torch import nn
from transformers import GPT2LMHeadModel

# Consider changing this to "cuda" for Nvidia GPUs or "mps" for Apple Metal GPUs
DEVICE = "cpu"
MODEL_ARCH_FILE = "examples/gpt2_xl.yml"
HF_MODEL_NAME = "gpt2-xl"


class NewGELU(nn.Module):
    def forward(self, x):
        c = math.sqrt(2.0 / math.pi)
        x_cubed = torch.pow(x, 3.0)

        return 0.5 * x * (1.0 + torch.tanh(c * (x + 0.044715 * x_cubed)))


class GPT2FeedForward(nn.Module):
    def __init__(self, config: FeedForwardConfig):
        super().__init__()

        self.in_dim = config.embedding_dimension
        self.hidden_dim = config.intermediate_dimension

        self.c_fc = nn.Linear(self.in_dim, self.hidden_dim)
        self.activation_fn = config.activation_fn
        self.c_proj = nn.Linear(self.hidden_dim, self.in_dim)
        self.dropout = nn.Dropout(config.dropout)

        self.config = config

    def forward(self, x):
        x = self.c_fc(x)
        x = self.activation_fn(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class GPT2(Transformer):
    def prompt(self, prompt):
        enc = tiktoken.encoding_for_model("gpt2")

        input_ids = enc.encode(prompt)
        input_ids = torch.tensor(input_ids, dtype=torch.long, device=DEVICE).unsqueeze(
            0
        )

        start = time.monotonic()
        output = self.generate(input_ids, top_k=2, max_new_tokens=20, temperature=0.9)
        end = time.monotonic()

        output = enc.decode(output.flatten().tolist())
        elapsed_time = end - start

        print(f"Output: {output}")
        print(f"Response time: {elapsed_time:.2f}s")


def load_from_state_dict(original_state_dict):
    # Naming conventions between gpt2 and this framework differ
    name_changes = [
        (".h.", ".blocks."),
        ("ln_1.", "attn_norm."),
        ("ln_2.", "mlp_norm."),
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

    transformer_config = TransformerConfig.from_yml(MODEL_ARCH_FILE)
    transformer_config.ffn = GPT2FeedForward
    transformer_config.ffn.activation_fn = NewGELU()

    gpt2_xl = GPT2(transformer_config)
    gpt2_xl.load_state_dict(gpt2_xl.state_dict() | new_state_dict)

    gpt2_xl.to(DEVICE)
    gpt2_xl.eval()

    gpt2_xl.prompt("Claude Shannon, the")


if __name__ == "__main__":
    main()
