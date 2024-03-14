import time
import torch
import gc

from lm_builder.transformer import Transformer, TransformerConfig
from lm_builder.utils import change_state_dict_names, merge_state_dict_qkv_projs

from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()

# Consider changing this to "cuda" for Nvidia GPUs or "mps" for Apple Metal GPUs
DEVICE = "cpu"
MODEL_ARCH_FILE = "examples/llama2_7b_chat.yml"
HF_MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"

CONFIG = TransformerConfig.from_yml(MODEL_ARCH_FILE)


class Llama2(Transformer):
    def prompt(self, text):
        prompt = f"[INST] {text} [/INST]"
        enc = AutoTokenizer.from_pretrained(HF_MODEL_NAME)

        input_ids = enc.encode(prompt)
        input_ids = torch.tensor(input_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)

        start = time.monotonic()
        output = self.generate(
            input_ids, output_only=True, top_k=2, max_new_tokens=20, temperature=0.9
        )
        end = time.monotonic()

        output = enc.decode(output.flatten().tolist())
        elapsed_time = end - start

        print(f"Output: {output}")
        print(f"Response time: {elapsed_time:.2f}s")


def unpermute(weight: torch.Tensor):
    # When loading Llama 2 through Huggingface we need to change the view of the query and
    # key matrices because Huggingface themselves changes these matrices to make their RoPE
    # easier to compute. We do not implement the same logic for RoPE so we undo this.

    attn_config = CONFIG.attention_config
    n_heads = attn_config.num_heads
    emb_dim = attn_config.embedding_dimension

    weight = weight.view(n_heads, 2, emb_dim // n_heads // 2, emb_dim)
    return weight.transpose(1, 2).reshape(emb_dim, emb_dim)


def load_from_state_dict(original_state_dict):
    for i in range(CONFIG.num_layers):
        prefix = f"model.layers.{i}.self_attn."

        q_proj = prefix + "q_proj.weight"
        k_proj = prefix + "k_proj.weight"
        v_proj = prefix + "v_proj.weight"
        qkv_proj = prefix + "qkv_proj.weight"

        original_state_dict[q_proj].copy_(unpermute(original_state_dict[q_proj]))
        original_state_dict[k_proj].copy_(unpermute(original_state_dict[k_proj]))

        # Llama 2 have individual nn.Linear layers for each q, k, and v matrix so we have to
        # merge them for our implementation
        original_state_dict = merge_state_dict_qkv_projs(
            original_state_dict, q_proj, k_proj, v_proj, qkv_proj
        )

    name_changes = [
        ("model.", "transformer."),
        (".layers.", ".blocks."),
        ("input_layernorm", "attn_norm"),
        ("rotary_emb", "pos_emb"),
        ("o_proj.", "out_proj."),
        ("self_attn.", "attn."),
        ("post_attention_layernorm", "mlp_norm"),
        ("embed_tokens.", "wte."),
    ]

    return change_state_dict_names(original_state_dict, name_changes)


def main():
    # Load llama2-7b-chat model from huggingface to get the state dict
    model_hf = AutoModelForCausalLM.from_pretrained(HF_MODEL_NAME, device_map=DEVICE)
    state_dict = model_hf.state_dict()

    # Get a state dict with our naming conventions
    new_state_dict = load_from_state_dict(state_dict)
    del state_dict
    del model_hf
    gc.collect()

    llama2_7b = Llama2(CONFIG)
    llama2_7b.to(DEVICE)

    llama2_7b.load_state_dict(new_state_dict)
    del new_state_dict
    gc.collect()

    llama2_7b.eval()

    llama2_7b.prompt("Who is Claude Shannon?")


if __name__ == "__main__":
    main()
