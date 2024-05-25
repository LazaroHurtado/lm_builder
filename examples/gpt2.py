import math
import time
import torch
import torch.nn as nn
import gc

from lm_builder.ffn import FeedForwardConfig
from lm_builder.transformer import Transformer, TransformerConfig
from lm_builder.utils import change_state_dict_names

from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "cpu"
MODEL_ARCH_FILE = "examples/gpt2_xl.yml"
HF_MODEL_NAME = "openai-community/gpt2-xl"

class NewGELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

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
    def __init__(self, config: TransformerConfig):
        super().__init__(config)
        self.tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
    
    def prompt(self, prompt):
        input_ids = self.tokenizer(prompt, return_tensors="pt").to(DEVICE)["input_ids"]

        start = time.monotonic()
        output = self.generate(input_ids, max_new_tokens=20, temperature=0.4)
        end = time.monotonic()

        output = self.tokenizer.batch_decode(output)[0]
        elapsed_time = end - start
        
        print(f"Output: {output}")
        print(f"Response time: {elapsed_time:.2f}s")

def load_from_state_dict(original_state_dict):
    # Naming conventions between gpt2 and this framework differ
    name_changes = [(".h.", ".blocks."),
                    ("attn.c_proj", "attn.out_proj"),
                    (".ln_1.", ".attn_norm."),
                    (".ln_2.", ".mlp_norm."),
                    (".ln_f.", ".norm.")]
    # GPT2 used convolutions instead of linear modules so these weights won't match
    # our linear layer weights, instead we have to transpose them.
    to_transpose = ["attn.c_attn.weight",
                    "attn.c_proj.weight",
                    "mlp.c_fc.weight",
                    "mlp.c_proj.weight"]
    
    to_partition = ".c_attn."

    return change_state_dict_names(original_state_dict,
                                   name_changes,
                                   to_transpose,
                                   to_partition)

def main():
    # Load gpt2-xl model from huggingface to get the state dict
    model_hf = AutoModelForCausalLM.from_pretrained(HF_MODEL_NAME)
    state_dict = model_hf.state_dict()

    new_state_dict = load_from_state_dict(state_dict)
    del state_dict, model_hf
    gc.collect()

    transformer_config = TransformerConfig.from_yml(MODEL_ARCH_FILE)
    transformer_config.ffn = GPT2FeedForward
    transformer_config.ffn.activation_fn = NewGELU()
    
    with torch.no_grad():
        gpt2_xl = GPT2(transformer_config)
        gpt2_xl.load_state_dict(new_state_dict)

    del new_state_dict
    gc.collect()

    gpt2_xl.to(DEVICE)
    gpt2_xl.eval()

    gpt2_xl.prompt("Claude Shannon, the")

if __name__ == "__main__":
    main()