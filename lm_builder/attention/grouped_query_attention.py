from __future__ import annotations

from torch import nn

from .config import AttentionConfig
from .multi_query_attention import MultiQueryAttention


class GroupedQueryAttention(MultiQueryAttention):
    # GroupedQueryAttention (GQA) is similar to MultiQueryAttention (MQA) but
    # instead of having a single key and value head that is shared across all
    # query heads, we have multiple which are shared. For example, in MQA we have
    # 1 key and value head that is shared across 8 query heads, but in GQA we could
    # have 4 key and value heads where each head is shared with 2 of the 8 query heads.

    def __init__(self, config: AttentionConfig):
        super().__init__(config)

        self.kv_heads = config.kv_heads

        assert (
            self.num_heads % self.kv_heads == 0
        ), "Number of query heads must be divisible by the number of key/value heads."

        self.shared_heads = self.num_heads // self.kv_heads
        self.kv_dim = self.kv_heads * self.head_dim

        self.q_proj = nn.Linear(self.embedding_dim, self.q_dim, bias=config.bias)
        self.k_proj = nn.Linear(self.embedding_dim, self.kv_dim, bias=config.bias)
        self.v_proj = nn.Linear(self.embedding_dim, self.kv_dim, bias=config.bias)
