import torch.nn as nn

from ..attention.attention import Attention
from ..ffn.feed_forward import FeedForward

class Block(nn.Module):
    
    def __init__(self, attn_mechanism: Attention, ffn: FeedForward):
        super().__init__()
        
        self.attn_config = attn_mechanism.config
        self.ffn_config = ffn.config

        self.embedding_dim = self.attn_config.embedding_dimension

        self.ln_1 = nn.LayerNorm(self.embedding_dim)
        self.attn = attn_mechanism
        
        self.ln_2 = nn.LayerNorm(self.embedding_dim)
        self.mlp = ffn
    
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        
        return x