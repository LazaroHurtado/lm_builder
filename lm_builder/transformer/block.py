import torch
from torch import nn

from .config import TransformerConfig


class Block(nn.Module):

    def __init__(self, config: TransformerConfig, attention_type=None):
        super().__init__()

        self.embedding_dim = config.attention_config.embedding_dimension

        self.attn_norm = config.attn_norm(self.embedding_dim, bias=config.norm_bias)
        attention_type = attention_type or config.attention
        self.attn = attention_type(config.attention_config)

        self.ffn_norm = config.ffn_norm(self.embedding_dim, bias=config.norm_bias)
        self.ffn = config.ffn(config.ffn_config)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask=None,
        position_ids=None,
        kv_cache=None,
    ):
        x = x + self.attn(
            self.attn_norm(x),
            attention_mask=attention_mask,
            position_ids=position_ids,
            kv_cache=kv_cache,
        )
        x = x + self.ffn(self.ffn_norm(x))

        return x
