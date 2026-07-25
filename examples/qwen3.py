import gc
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from lm_builder import TextGenerationPipeline
from lm_builder.transformer import Transformer, TransformerConfig
from lm_builder.utils import (
    change_state_dict_names,
    combine_qkv_projections,
    get_device,
)


class Qwen3Loader:
    MODEL_ARCH_FILE = "examples/qwen3_0_6b.yml"
    HF_MODEL_NAME = "Qwen/Qwen3-0.6B"
    WEIGHTS_FILE = "qwen3_0_6b_weights.pth"

    def build_state_dict(self):
        model_hf = AutoModelForCausalLM.from_pretrained(
            self.HF_MODEL_NAME,
            device_map="cpu",
            torch_dtype="auto",
        )
        state_dict = model_hf.state_dict()

        new_state_dict = self.convert_state_dict(state_dict)
        del model_hf, state_dict
        gc.collect()

        torch.save(new_state_dict, self.WEIGHTS_FILE)
        del new_state_dict
        gc.collect()

    def build_model(self, rank):
        transformer_config = TransformerConfig.from_yml(self.MODEL_ARCH_FILE)

        with torch.no_grad():
            qwen3 = Transformer(transformer_config)
            qwen3.to(rank)
            qwen3.eval()
        return qwen3

    def convert_state_dict(self, original_state_dict):
        name_changes = [
            (".mlp.", ".ffn."),
            (".layers.", ".blocks."),
            ("input_layernorm", "attn_norm"),
            ("rotary_emb", "pos_emb"),
            ("o_proj", "out_proj"),
            ("self_attn", "attn"),
            ("post_attention_layernorm", "ffn_norm"),
            (".embed_tokens.weight", ".wte.weight"),
            ("model.", ""),
        ]

        state_dict = change_state_dict_names(original_state_dict, name_changes)
        return combine_qkv_projections(state_dict)


def main():
    loader = Qwen3Loader()
    if not os.path.exists(loader.WEIGHTS_FILE):
        print("Building Qwen3 state dict...")
        loader.build_state_dict()

    device = get_device()
    with torch.no_grad():
        qwen3 = loader.build_model("meta")
        state_dict = torch.load(loader.WEIGHTS_FILE, map_location="cpu")
        qwen3.load_state_dict(state_dict, assign=True)

        del state_dict
        gc.collect()

        messages = [
            {"role": "user", "content": "Who is Claude Shannon?"},
        ]
        generation_config = GenerationConfig.from_pretrained(loader.HF_MODEL_NAME)

        qwen3.to(device)
        qwen3 = torch.compile(qwen3, fullgraph=True)
        tokenizer = AutoTokenizer.from_pretrained(loader.HF_MODEL_NAME)
        pipeline = TextGenerationPipeline(qwen3, tokenizer)
        pipeline.prompt(
            messages,
            max_new_tokens=4096,
            temperature=generation_config.temperature,
            top_k=generation_config.top_k,
            top_p=generation_config.top_p,
            eos_token_id=generation_config.eos_token_id,
            apply_chat_template=True,
            stream=True,
            debug=True,
            device=device,
            use_cache=True,
        )


if __name__ == "__main__":
    main()
