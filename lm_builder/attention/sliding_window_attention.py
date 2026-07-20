from __future__ import annotations

import torch

from .casual_attention import CausalMultiHeadAttention
from .config import AttentionConfig


class SlidingWindowAttention(CausalMultiHeadAttention):

    def __init__(self, config: AttentionConfig):
        if (
            not isinstance(config.window_size, int)
            or isinstance(config.window_size, bool)
            or config.window_size <= 0
        ):
            raise ValueError("window_size must be a positive integer.")

        self.window_size = config.window_size
        super().__init__(config)

    def _register_mask(self):
        self.register_buffer(
            "attention_mask",
            torch.ones(self.context_len, self.context_len)
            .tril()
            .triu(diagonal=1 - self.window_size)[None, None, :, :],
            persistent=False,
        )
