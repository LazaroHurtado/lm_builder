from __future__ import annotations

import torch

from .config import AttentionConfig
from .multi_head_attention import MultiHeadAttention


class CausalMultiHeadAttention(MultiHeadAttention):

    def __init__(self, config: AttentionConfig):
        super().__init__(config)

    def _register_mask(self):
        # Causal attention allows tokens to attend to only
        # previous tokens, token t_i can also look at
        # tokens t_{0:i-1}. This mask is applied to the
        # Q*K^T matrix, which has a size of (B, T, T) so
        # we construct a matrix where elements above the
        # principle diagonal are zero with the same shape.
        self.register_buffer(
            "attention_mask",
            torch.ones(self.context_len, self.context_len).tril()[
                None, None, :, :
            ],
            persistent=False,
        )
