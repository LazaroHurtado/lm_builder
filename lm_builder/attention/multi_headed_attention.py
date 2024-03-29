from __future__ import annotations

import math
import torch

from .attention import Attention
from .config import AttentionConfig
from .kv_cache import KVCache

from torch import nn
from torch.nn import functional as F


class MultiHeadAttention(nn.Module, Attention):

    def __init__(self, config: AttentionConfig):
        super().__init__()
        assert config.embedding_dimension % config.num_heads == 0

        self.context_len = config.context_length
        self.embedding_dim = config.embedding_dimension
        self.num_heads = config.num_heads
        self.with_kv_cache = config.with_kv_cache

        self.head_dim = self.embedding_dim // self.num_heads

        self.qkv_proj = nn.Linear(
            self.embedding_dim, 3 * self.embedding_dim, bias=config.bias
        )
        self.out_proj = nn.Linear(
            self.embedding_dim, self.embedding_dim, bias=config.bias
        )
        self.attn_dropout = nn.Dropout(config.attn_dropout)
        self.resid_dropout = nn.Dropout(config.resid_dropout)

        if self.with_kv_cache:
            self.kv_cache = KVCache(self.context_len)

        self.has_positional_embedding = config.positional_embedding is not None
        if self.has_positional_embedding:
            self.pos_emb = config.positional_embedding(
                self.context_len, self.head_dim, config.pos_emb_freq
            )

        self.has_flash_attn = hasattr(F, "scaled_dot_product_attention")

        self.register_buffer(
            "attention_mask",
            torch.ones(self.context_len, self.context_len)[None, None, :, :],
            persistent=False,
        )

        self._config = config

    def get_qkv(self, x: torch.Tensor):
        # x has dimensionality of (batch_size, sequence_length, embedding_dim).
        # (B, T, C) -> (B, T, 3*C)
        qkv = self.qkv_proj(x)

        # (B, T, 3*C) -> (B, T, C), (B, T, C), (B, T, C)
        q, k, v = qkv.split(self.embedding_dim, dim=2)

        return q, k, v

    def get_heads(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        B, T, _ = query.size()  # pylint: disable=invalid-name

        # For the key and value matrices we might have hit the KV-Cache, so
        # its sequence length might not be T which is why we do `.size(dim=1)`
        # (B, T, C) -> (B, T, num_head, head_dim)
        q_heads = query.view(B, T, self.num_heads, self.head_dim)
        k_heads = key.view(B, key.size(dim=1), self.num_heads, self.head_dim)
        v_heads = value.view(B, value.size(dim=1), self.num_heads, self.head_dim)

        return q_heads, k_heads, v_heads

    def attention(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        B, T, _ = query.size()  # pylint: disable=invalid-name

        # (B, num_heads, T, head_dim) @ (B, num_heads, head_dim, T) -> (B, num_heads, T, T)
        scale = 1.0 / math.sqrt(key.size(dim=-1))
        attn = (query @ key.transpose(-2, -1)) * scale
        attn = attn.view(B, self.num_heads, T, T)

        # Apply mask to Q@K^T matrix. Where the mask is equal
        # to zero we will replace the matrix's element at
        # that position with -inf. We use -inf so when we apply
        # softmax those elements will equal to 0.
        attn = attn.masked_fill(self.attention_mask[:, :, :T, :T] == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)

        attn = self.attn_dropout(attn)

        # (B, num_heads, T, T) x (B, num_heads, T, head_dim) -> (B, num_heads, T, head_dim)
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
        B, T, C = x.size()  # pylint: disable=invalid-name

        # we get the q, k, v projection of each embedding, each
        # matrix will have dimension (B, T, C)
        q, k, v = self.get_qkv(x)

        # next we split the projected embeddings across the number
        # of heads we have, allowing each head to gain a different
        # interpretation.
        # (B, T, num_heads, head_dim)
        q, k, v = self.get_heads(q, k, v)
        if self.has_positional_embedding:
            q = self.pos_emb(q)
            k = self.pos_emb(k)
        
        if self.with_kv_cache and not self.training:
            k, v = self.kv_cache.update(k, v)

        # (B, num_heads, T, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.has_flash_attn:
            attn = F.scaled_dot_product_attention(  # pylint: disable=not-callable
                q,
                k,
                v,
                attn_mask=(self.attention_mask[:, :, :T, :T] == 1),
                dropout_p=self.attn_dropout.p,
            )
        else:
            attn = self.attention(q, k, v)

        # Convert multi-headed shaped matrix back to original shape
        # (B, num_heads, T, head_dim) -> (B, T, num_heads, head_dim)
        # -> (B*T*num_heads*head_dim) -> (B, T, C)
        attn = attn.transpose(1, 2).contiguous().view(B, T, C)

        out = self.out_proj(attn)
        out = self.resid_dropout(out)

        return out
