import gc
import os
from functools import partial

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from lm_builder import LanguageModel
from lm_builder.transformer import TransformerConfig
from lm_builder.utils import change_state_dict_names


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class GPT2Loader:
    MODEL_ARCH_FILE = "examples/gpt2_xl.yml"
    HF_MODEL_NAME = "openai-community/gpt2-xl"
    WEIGHTS_FILE = "gpt2_weights.pth"

    @staticmethod
    def build_state_dict():
        # Load gpt2-xl model from huggingface to get the state dict
        model_hf = AutoModelForCausalLM.from_pretrained(GPT2Loader.HF_MODEL_NAME)
        state_dict = model_hf.state_dict()

        new_state_dict = GPT2Loader.convert_state_dict(state_dict)
        del state_dict, model_hf
        gc.collect()

        torch.save(new_state_dict, GPT2Loader.WEIGHTS_FILE)
        del new_state_dict
        gc.collect()

    @staticmethod
    def build_model(rank):
        transformer_config = TransformerConfig.from_yml(GPT2Loader.MODEL_ARCH_FILE)
        transformer_config.ffn_config.activation_fn = partial(
            nn.GELU,
            approximate="tanh",
        )

        tokenizer = AutoTokenizer.from_pretrained(
            GPT2Loader.HF_MODEL_NAME, clean_up_tokenization_spaces=True
        )

        with torch.no_grad():
            gpt2_xl = LanguageModel(transformer_config, tokenizer)
            gpt2_xl.to(rank)
            gpt2_xl.eval()
        return gpt2_xl

    @staticmethod
    def convert_state_dict(original_state_dict):
        # Naming conventions between gpt2 and this framework differ
        name_changes = [
            (".h.", ".blocks."),
            ("mlp.c_fc", "ffn.up_proj"),
            ("mlp.c_proj", "ffn.down_proj"),
            ("ln_1.", "attn_norm."),
            ("ln_2.", "ffn_norm."),
            ("ln_f.", "norm."),
            ("attn.c_attn", "attn.qkv_proj"),
            ("attn.c_proj", "attn.out_proj"),
            ("transformer.", ""),
        ]
        # GPT2 used convolutions instead of linear modules so these weights won't match
        # our linear layer weights, instead we have to transpose them.
        to_transpose = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]

        return change_state_dict_names(
            original_state_dict,
            name_changes,
            to_transpose,
        )


def main():
    if not os.path.exists(GPT2Loader.WEIGHTS_FILE):
        print("Building GPT-2 state dict...")
        GPT2Loader.build_state_dict()

    with torch.no_grad():
        gpt2_xl = GPT2Loader.build_model("meta")
        state_dict = torch.load(GPT2Loader.WEIGHTS_FILE, map_location="cpu")
        gpt2_xl.load_state_dict(state_dict, assign=True)

        del state_dict
        gc.collect()

        gpt2_xl.to(get_device())
        prompt = "Claude Shannon, the"
        generation_kwargs = {
            "max_new_tokens": 200,
            "temperature": 0.7,
            "stream": True,
            "debug": True,
            "device": get_device(),
        }

        print("Generating with KV cache:")
        print(prompt, end="", flush=True)
        gpt2_xl.prompt(
            prompt,
            use_cache=True,
            **generation_kwargs,
        )

        print("\nGenerating without KV cache:")
        print(prompt, end="", flush=True)
        gpt2_xl.prompt(
            prompt,
            use_cache=False,
            **generation_kwargs,
        )


if __name__ == "__main__":
    main()
