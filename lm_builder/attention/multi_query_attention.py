from __future__ import annotations

import torch
import torch.nn as nn

from .config import AttentionConfig
from .casual_attention import CausalMultiHeadAttention

class MultiQueryAttention(CausalMultiHeadAttention):

    def __init__(self, config: AttentionConfig):
        super().__init__(config)

        self.kv_heads = 1
        self.shared_heads = self.num_heads // self.kv_heads

        self.q_dim = self.embedding_dim
        self.kv_dim = self.head_dim

        self.qkv_proj = nn.Linear(self.embedding_dim, self.q_dim + 2*self.kv_dim)
    
    def get_qkv(self, x: torch.Tensor):
        # x has dimensionality of (batch_size, sequence_length, embedding_dim).
        # (B, T, C) -> (B, T, q_dim+2*kv_dim)
        qkv = self.qkv_proj(x)
        
        # (B, T, q_dim+2*kv_dim) -> (B, T, C), (B, T, head_dim), (B, T, head_dim)
        q, k, v = qkv.split([self.q_dim, self.kv_dim, self.kv_dim], dim = 2)

        if self.with_kv_cache and not self.training:
            k, v = self.kv_cache.update(k ,v)

        return q, k, v

    def get_heads(self,
                  query: torch.Tensor,
                  key: torch.Tensor,
                  value: torch.Tensor):
        # In multi-query attention (MQA), the query is the only tensor
        # that is split into multiple heads, the key and value are
        # only one head size. This affects performance but it reduces
        # the memory transfer latency of the GPU, especially for long
        # sequences. During inference, we can see up to 7x speedup.
        
        B, T, _ = query.size()
        # (B, T, C) -> (B, T, num_head, head_dim) -> (B, num_heads, T, head_dim)
        q_heads = query.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # We need to consider what would happen if kv_heads > 1, in the case of
        # GroupedQueryAttention calling this method. In that case, we can't multiply
        # Q and K^T because the dimensions at dim=1 would not work. Instead, we need
        # to split the query heads into groups of kv_heads.
        # Example:
        #   Let Q be of dimension (1, num_heads=8, T=10, head_dim=5) and
        #   K of dimension (B, kv_heads=2, T=10, head_dim=5).
        #   Doing Q@K^T would give us an error because num_heads != kv_heads.
        #   We can get around this by converting Q to (1, 4, 2, 10, 5). Then,
        #   after doing Q@K^T, we have to remembering to convert Q back to its
        #   original shape, (1, 8, 10, 5).
        # (B, num_heads, T, head_dim) -> (B, num_heads, 1, T, head_dim)
        q_heads = q_heads.view(B, self.shared_heads, self.kv_heads, T, self.head_dim)
        
        # (B, T, head_dim) -> (B, T, 1, head_dim) -> (B, 1, T, head_dim)
        k_heads = key.view(B, key.size(dim=1), self.kv_heads, self.head_dim).transpose(1, 2)
        v_heads = value.view(B, value.size(dim=1), self.kv_heads, self.head_dim).transpose(1, 2)

        return q_heads, k_heads, v_heads