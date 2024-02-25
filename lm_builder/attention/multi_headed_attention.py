from __future__ import annotations

import math
import torch
import torch.nn as nn

from .attention import Attention
from .config import AttentionConfig
from .kv_cache import KVCache

from torch.nn import functional as F

class MultiHeadAttention(nn.Module, Attention):

    def __init__(self, config: AttentionConfig, with_mask=True):
        super().__init__()
        assert config.embedding_dimension % config.num_heads == 0
        
        self.context_len = config.context_length
        self.embedding_dim = config.embedding_dimension
        self.num_heads = config.num_heads
        self.with_kv_cache = config.with_kv_cache
        
        self.qkv_proj = nn.Linear(self.embedding_dim, 3*self.embedding_dim)
        self.out_proj = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.attn_dropout = nn.Dropout(config.attn_dropout)
        self.resid_dropout = nn.Dropout(config.resid_dropout)

        if self.with_kv_cache:
            self.kv_cache = KVCache(self.context_len)
        
        self.has_flash_attn = hasattr(F, 'scaled_dot_product_attention')
        
        if with_mask:
            self.register_buffer(
                "attention_mask",
                torch.ones(self.context_len, self.context_len)
                    .view(1, 1, self.context_len, self.context_len),
                persistent=False)

        self._config = config
    
    def get_qkv(self, x):
        # x has dimensionality of (batch_size, sequence_length, embedding_dim).
        # (B, T, C) -> (B, T, 3*C)
        qkv = self.qkv_proj(x)
        
        # (B, T, 3*C) -> (B, T, C), (B, T, C), (B, T, C)
        q, k, v = qkv.split(self.embedding_dim, dim = 2)

        if self.with_kv_cache and not self.training:
            k, v = self.kv_cache.update(k ,v)

        return q, k, v

    def attention(self, query, key, value):
        T = query.size(dim=2)
        
        # (B, num_heads, T, head_size) x (B, num_heads, head_size, T) -> (B, num_heads, T, T)
        scale = (1.0 / math.sqrt(key.size(dim=-1)))
        attn = (query @ key.transpose(-2, -1)) * scale
        
        # Apply mask to Q@K^T matrix. Where the mask is equal
        # to zero we will replace the matrix's element at
        # that position with -inf. We use -inf so when we apply
        # softmax those elements will equal to 0.
        attn = attn.masked_fill(self.attention_mask[:, :, :T, :T] == 0, float('-inf'))
        attn = F.softmax(attn, dim = -1)
        
        attn = self.attn_dropout(attn)
        
        # (B, num_heads, T, T) x (B, num_heads, T, head_size) -> (B, num_heads, T, head_size)
        return attn @ value
    
    def forward(self, x):
        if self.with_kv_cache and not self.training:
            # If we are using kv cache then we only care about the last token
            # since we are caching previous tokens in the K and V matrices. The
            # only exception is in the beginning when we need to populate the
            # cache with the K and V values from the prompt, this is only done
            # once.
            if not self.kv_cache.first_prefill:
                # (B, T, C) -> (B, 1, C)
                x = x[:, -1:, :]
            self.kv_cache.first_prefill = False

        # batch size, sequence length, embedding dimensionality.
        # Here, T might be 1 if we are using kv_cache during inference.
        B, T, C = x.size()
        
        # we get the q, k, v projection of each embedding, each
        # matrix will have dimension (B, T, C)
        q, k, v = self.get_qkv(x)

        # next we split the projected embeddings across the number
        # of heads we have, allowing each head to gain a different
        # interpretation.
        # (B, num_heads, T, head_size)
        head_size = C // self.num_heads
        q = q.view(B, T, self.num_heads, head_size).transpose(1, 2)
        k = k.view(B, k.size(dim=1), self.num_heads, head_size).transpose(1, 2)
        v = v.view(B, v.size(dim=1), self.num_heads, head_size).transpose(1, 2)
        
        if self.has_flash_attn:
            attn = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask = (self.attention_mask[:, :, :T, :T] == 1),
                dropout_p = self.attn_dropout.p)
        else:
            attn = self.attention(q, k, v)
        
        # Convert multi-headed shaped matrix back to original shape
        # (B, num_heads, T, head_size)  -> (B, T, num_heads, head_size)
        # -> (B*T*num_heads*head_size)  -> (B, T, C)
        attn = attn.transpose(1, 2).contiguous().view(B, T, C)
        
        out = self.out_proj(attn)
        out = self.resid_dropout(out)
        
        return out