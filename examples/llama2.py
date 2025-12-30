import gc
import os

import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

from lm_builder import LanguageModel
from lm_builder.transformer import TransformerConfig
from lm_builder.utils import change_state_dict_names

load_dotenv()

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"


class Llama2Loader:
    MODEL_ARCH_FILE = "examples/llama2_7b_chat.yml"
    HF_MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
    WEIGHTS_FILE = "llama2_weights.pth"

    def build_state_dict(self):
        # Load llama2-7b-chat model from huggingface to get the state dict
        model_hf = AutoModelForCausalLM.from_pretrained(
            self.HF_MODEL_NAME, device_map="cpu"
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
        tokenizer = AutoTokenizer.from_pretrained(self.HF_MODEL_NAME)

        with torch.no_grad():
            llama2_7b = LanguageModel(transformer_config, tokenizer)
            llama2_7b.to(rank)
            llama2_7b.eval()
        return llama2_7b

    def convert_state_dict(self, original_state_dict):
        name_changes = [
            ("model.", "transformer."),
            (".mlp.", ".ffn."),
            (".layers.", ".blocks."),
            ("input_layernorm", "attn_norm"),
            ("rotary_emb", "pos_emb"),
            ("o_proj", "out_proj"),
            ("self_attn", "attn"),
            ("post_attention_layernorm", "ffn_norm"),
            (".embed_tokens.weight", ".wte.weight"),
        ]

        return change_state_dict_names(original_state_dict, name_changes)


def main():
    if not os.path.exists("llama2_weights.pth"):
        print("Building Llama2 state dict...")
        Llama2Loader().build_state_dict()

    with torch.no_grad():
        llama2_7b = Llama2Loader().build_model("meta")
        state_dict = torch.load(Llama2Loader.WEIGHTS_FILE, map_location=DEVICE)
        llama2_7b.load_state_dict(state_dict, assign=True)

        del state_dict
        gc.collect()

        messages = [
            {"role": "user", "content": "Who is Claude Shannon?"},
        ]

        llama2_7b.to(DEVICE)
        output = llama2_7b.prompt(
            messages,
            max_new_tokens=100,
            temperature=0,
            apply_chat_template=True,
            debug=True,
            device=DEVICE,
            stream=True,
        )
        print(output)


if __name__ == "__main__":
    main()
