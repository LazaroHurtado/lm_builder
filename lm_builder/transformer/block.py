import torch
from torch import nn

from ..attention import AttentionLayerConfig
from ..ffn import FeedForwardConfig


class Block(nn.Module):

    def __init__(
        self,
        attention_config: AttentionLayerConfig,
        ffn_config: FeedForwardConfig,
        qk_positional_embedding=None,
    ):
        super().__init__()

        self.embedding_dim = attention_config.embedding_dimension

        self.attn_norm = attention_config.norm.build(self.embedding_dim)
        self.attn = attention_config.attention_type(
            attention_config, qk_positional_embedding=qk_positional_embedding
        )

        self.ffn_norm = ffn_config.norm.build(self.embedding_dim)
        self.ffn = ffn_config.ffn_type(ffn_config)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask=None,
        qk_position_data=None,
        kv_cache=None,
    ):
        x = x + self.attn(
            self.attn_norm(x),
            attention_mask=attention_mask,
            qk_position_data=qk_position_data,
            kv_cache=kv_cache,
        )
        ffn_out, routing_loss = self.ffn(
            self.ffn_norm(x),
            token_mask=attention_mask,
        )
        x = x + ffn_out

        return x, routing_loss
