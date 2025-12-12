from __future__ import annotations

import torch
from torch import nn

from .casual_attention import CausalMultiHeadAttention
from .config import AttentionConfig


class MultiQueryAttention(CausalMultiHeadAttention):

    def __init__(self, config: AttentionConfig):
        super().__init__(config)

        self.kv_heads = 1
        self.shared_heads = self.num_heads // self.kv_heads

        self.q_dim = self.embedding_dim
        self.kv_dim = self.head_dim

        self.q_proj = nn.Linear(self.embedding_dim, self.q_dim, bias=config.bias)
        self.k_proj = nn.Linear(self.embedding_dim, self.kv_dim, bias=config.bias)
        self.v_proj = nn.Linear(self.embedding_dim, self.kv_dim, bias=config.bias)

    def get_qkv(self, x: torch.Tensor):
        # x has dimensionality of (batch_size, sequence_length, embedding_dim).
        # (B, T, C) -> (B, T, q_dim)
        q = self.q_proj(x)
        # (B, T, C) -> (B, T, kv_dim)
        k, v = self.k_proj(x), self.v_proj(x)

        return q, k, v

    def get_heads(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        # In multi-query attention (MQA), the query is the only tensor
        # that is split into multiple heads, the key and value are
        # only one head size. This affects performance but it reduces
        # the memory transfer latency of the GPU, especially for long
        # sequences. During inference, we can see up to 7x speedup.

        B, T, _ = query.size()
        # (B, T, C) -> (B, T, num_head, head_dim)
        q_heads = query.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # (B, T, head_dim) -> (B, T, 1, head_dim)
        k_heads = key.view(B, key.size(dim=1), self.kv_heads, self.head_dim).transpose(
            1, 2
        )
        v_heads = value.view(
            B, value.size(dim=1), self.kv_heads, self.head_dim
        ).transpose(1, 2)

        return q_heads, k_heads, v_heads
