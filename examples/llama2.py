import torch
import gc

from lm_builder import LanguageModel
from lm_builder.transformer import TransformerConfig
from lm_builder.positional_embeddings.rotary_pe import RotaryPE
from lm_builder.utils import change_state_dict_names

from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()

DEVICE = "cpu"
MODEL_ARCH_FILE = "examples/llama2_7b_chat.yml"
HF_MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"


class LlamaRoPE(RotaryPE):
    def forward(self, x: torch.Tensor, unsqueeze_dim=1):
        T = x.size(dim=-2)

        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        y = torch.cat((-x2, x1), dim=-1)

        cos = self._cos_cached[:, :T, :].unsqueeze(unsqueeze_dim)
        sin = self._sin_cached[:, :T, :].unsqueeze(unsqueeze_dim)
        return x * cos + y * sin


def load_from_state_dict(original_state_dict):
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
    # Load llama2-7b-chat model from huggingface to get the state dict
    model_hf = AutoModelForCausalLM.from_pretrained(HF_MODEL_NAME, device_map="cpu")
    state_dict = model_hf.state_dict()

    new_state_dict = load_from_state_dict(state_dict)
    del model_hf, state_dict
    gc.collect()

    transformer_config = TransformerConfig.from_yml(MODEL_ARCH_FILE)
    transformer_config.attention_config.positional_embedding = LlamaRoPE

    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)

    with torch.no_grad():
        tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
        llama2_7b = LanguageModel(transformer_config, tokenizer, device=DEVICE)
        llama2_7b.load_state_dict(new_state_dict)

    del new_state_dict
    gc.collect()

    llama2_7b.to(DEVICE)
    llama2_7b.eval()

    llama2_7b.prompt(
        "[INST] Who is Claude Shannon? [/INST]",
        output_only=True,
        top_k=2,
        max_new_tokens=20,
        temperature=0.9,
    )


if __name__ == "__main__":
    main()
