from __future__ import annotations

from .casual_attention import CausalMultiHeadAttention
from .config import AttentionConfig


class MultiQueryAttention(CausalMultiHeadAttention):

    def _get_num_kv_heads(self, config: AttentionConfig):
        # MQA shares one key and value head across every query head.
        return 1
