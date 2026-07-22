from __future__ import annotations

import torch

from .multi_head_attention import MultiHeadAttention


class CausalMultiHeadAttention(MultiHeadAttention):
    supports_kv_cache = True
    supports_window_size = True
    is_causal = True

    def _build_base_attention_mask(
        self,
        query_length,
        key_length,
        device,
    ):
        query_start = key_length - query_length
        query_positions = torch.arange(
            query_start,
            key_length,
            device=device,
        ).unsqueeze(-1)
        key_positions = torch.arange(key_length, device=device).unsqueeze(0)
        attention_mask = key_positions <= query_positions

        if self.window_size is not None:
            window_start = query_positions - self.window_size + 1
            attention_mask &= key_positions >= window_start

        return attention_mask[None, None, :, :]
