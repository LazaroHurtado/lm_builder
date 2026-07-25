from __future__ import annotations

from .casual_attention import CausalMultiHeadAttention
from .config import AttentionLayerConfig


class GroupedQueryAttention(CausalMultiHeadAttention):
    # GroupedQueryAttention (GQA) shares each key and value head across a group of
    # query heads. For example, 4 key/value heads across 8 query heads means each
    # key/value head is shared by 2 query heads. kv_heads=1 shares a single
    # key/value head across every query head, which is multi-query attention.

    def _get_num_kv_heads(self, config: AttentionLayerConfig):
        return config.kv_heads
