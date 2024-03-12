import time
import torch

from lm_builder.transformer import Transformer, TransformerConfig
from lm_builder.utils import change_state_dict_names, merge_state_dict_qkv_projs

from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()

MODEL_ARCH_FILE = "examples/llama2_7b_chat.yml"
HF_MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"

class Llama2(Transformer):
    def __init__(self, config: TransformerConfig, device=None):
        super().__init__(config)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    def load_from_state_dict(self, original_state_dict):
        model_state_dict = self.state_dict()

        original_state_dict = merge_state_dict_qkv_projs(original_state_dict, "q_proj", "k_proj", "v_proj", "qkv_proj")

        name_changes = [("model.", "transformer."),
                        (".layers.", ".blocks."),
                        ("input_layernorm", "attn_norm"),
                        ("rotary_emb", "pos_emb"),
                        ("o_proj", "out_proj"),
                        ("self_attn", "attn"),
                        ("post_attention_layernorm", "mlp_norm"),
                        (".embed_tokens", ".wte")]

        self.state_dict = change_state_dict_names(model_state_dict,
                                                  original_state_dict,
                                                  name_changes)
        self.to(self.device)
    
    def prompt(self, prompt):
        enc = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
        
        input_ids = enc.encode(prompt)
        input_ids = torch.tensor(input_ids, dtype=torch.long, device=self.device).unsqueeze(0)

        start = time.monotonic()
        output = self.generate(input_ids, max_new_tokens=20, temperature=0.4)
        end = time.monotonic()

        output = enc.decode(output.flatten().tolist())
        elapsed_time = end - start
        
        print(f"Output: {output}")
        print(f"Response time: {elapsed_time:.2f}s")

def main():
    transformer_config = TransformerConfig.from_yml(MODEL_ARCH_FILE)
    llama2_7b = Llama2(transformer_config)

    # Load llama2-7b-chat model from huggingface to get the state dict
    model_hf = AutoModelForCausalLM.from_pretrained(HF_MODEL_NAME)
    state_dict = model_hf.state_dict()

    # Load huggingface's state dict into our model
    llama2_7b.load_from_state_dict(state_dict)
    llama2_7b.eval()

    llama2_7b.prompt("Claude Shannon, the")

if __name__ == "__main__":
    main()