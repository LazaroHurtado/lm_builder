from __future__ import annotations

from .config import AttentionConfig
from .multi_query_attention import MultiQueryAttention


class GroupedQueryAttention(MultiQueryAttention):
    # GroupedQueryAttention (GQA) is similar to MultiQueryAttention (MQA) but
    # instead of having a single key and value head that is shared across all
    # query heads, we have multiple which are shared. For example, in MQA we have
    # 1 key and value head that is shared across 8 query heads, but in GQA we could
    # have 4 key and value heads where each head is shared with 2 of the 8 query heads.

    def _get_num_kv_heads(self, config: AttentionConfig):
        return config.kv_heads
