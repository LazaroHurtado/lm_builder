from .config import TransformerConfig

from torch import nn


class Block(nn.Module):

    def __init__(self, config: TransformerConfig):
        super().__init__()

        self.embedding_dim = config.attention_config.embedding_dimension

        self.attn_norm = config.attn_norm(self.embedding_dim, bias=config.norm_bias)
        self.attn = config.attention(config.attention_config)

        self.mlp_norm = config.mlp_norm(self.embedding_dim, bias=config.norm_bias)
        self.mlp = config.ffn(config.ffn_config)

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.mlp_norm(x))

        return x
