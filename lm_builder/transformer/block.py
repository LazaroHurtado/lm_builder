import torch
from torch import nn

from .config import TransformerConfig


class Block(nn.Module):

    def __init__(self, config: TransformerConfig):
        super().__init__()

        self.embedding_dim = config.attention_config.embedding_dimension

        self.attn_norm = config.attn_norm(self.embedding_dim, bias=config.norm_bias)
        self.attn = config.attention(config.attention_config)

        self.ffn_norm = config.ffn_norm(self.embedding_dim, bias=config.norm_bias)
        self.ffn = config.ffn(config.ffn_config)

    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))

        return x
