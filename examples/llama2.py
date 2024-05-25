import time
import torch
import torch.nn as nn
import gc

from lm_builder.transformer import Transformer, TransformerConfig
from lm_builder.positional_embeddings.rotary_pe import RotaryPE
from lm_builder.utils import change_state_dict_names

from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()

DEVICE = "cpu"
MODEL_ARCH_FILE = "examples/llama2_7b_chat.yml"
HF_MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"

class RMSNorm(nn.Module):
    def __init__(self, dim: int, **kwargs):
        super().__init__()
        self.eps = 1e-5
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class LlamaRoPE(RotaryPE):
    def forward(self, x: torch.Tensor, unsqueeze_dim=1):
        T = x.size(dim=-2)

        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        y = torch.cat((-x2, x1), dim=-1)

        cos = self._cos_cached[:, :T, :].unsqueeze(unsqueeze_dim)
        sin = self._sin_cached[:, :T, :].unsqueeze(unsqueeze_dim)
        return x * cos + y * sin

class Llama2(Transformer):
    def __init__(self, config: TransformerConfig):
        super().__init__(config)
        self.tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
    
    def prompt(self, prompt):
        prompt = f"[INST] <<SYS>>\nYou are a helpful assistant that answers all questions\n<</SYS>>\n\n{ prompt } [/INST]"
        input_ids = self.tokenizer(prompt, return_tensors="pt").to(DEVICE)["input_ids"]

        start = time.monotonic()
        output = self.generate(input_ids, output_only=True, max_new_tokens=20, temperature=0.4)
        end = time.monotonic()

        output = self.tokenizer.batch_decode(output)[0]
        elapsed_time = end - start
        
        print(f"Output: {output}")
        print(f"Response time: {elapsed_time:.2f}s")

def load_from_state_dict(original_state_dict):
    name_changes = [("model.", "transformer."),
                    (".layers.", ".blocks."),
                    ("input_layernorm", "attn_norm"),
                    ("rotary_emb", "pos_emb"),
                    ("o_proj", "out_proj"),
                    ("self_attn", "attn"),
                    ("post_attention_layernorm", "mlp_norm"),
                    (".embed_tokens.weight", ".wte.weight")]

    return change_state_dict_names(original_state_dict,
                                   name_changes)

def main():
    # Load llama2-7b-chat model from huggingface to get the state dict
    model_hf = AutoModelForCausalLM.from_pretrained(HF_MODEL_NAME, device_map="cpu")
    state_dict = model_hf.state_dict()

    new_state_dict = load_from_state_dict(state_dict)
    del model_hf, state_dict
    gc.collect()

    transformer_config = TransformerConfig.from_yml(MODEL_ARCH_FILE)
    transformer_config.attn_norm = RMSNorm
    transformer_config.mlp_norm = RMSNorm
    transformer_config.transformer_norm = RMSNorm
    transformer_config.attention_config.positional_embedding = LlamaRoPE

    with torch.no_grad():
        llama2_7b = Llama2(transformer_config)
        llama2_7b.load_state_dict(new_state_dict)

    del new_state_dict
    gc.collect()

    llama2_7b.to(DEVICE)
    llama2_7b.eval()

    llama2_7b.prompt("Who is Claude Shannon?")

if __name__ == "__main__":
    main()