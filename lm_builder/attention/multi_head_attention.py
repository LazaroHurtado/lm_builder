from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from ..inference import KVCache
from .attention import Attention
from .config import AttentionLayerConfig


class MultiHeadAttention(Attention):
    supports_kv_cache = False
    supports_window_size = False
    is_causal = False

    def __init__(
        self,
        config: AttentionLayerConfig,
        qk_positional_embedding=None,
    ):
        super().__init__()
        if config.window_size is not None and (
            not self.supports_window_size or not self.is_causal
        ):
            raise ValueError(f"{type(self).__name__} does not support window_size.")

        self.context_len = config.context_length
        self.embedding_dim = config.embedding_dimension
        self.num_heads = config.num_heads
        self.window_size = config.window_size
        self.qk_positional_embedding = qk_positional_embedding
        self.head_dim = config.head_dim

        self.kv_heads = self._get_num_kv_heads(config)
        assert (
            self.num_heads % self.kv_heads == 0
        ), "Number of query heads must be divisible by the number of key/value heads."
        self.shared_heads = self.num_heads // self.kv_heads

        self.q_dim = self.num_heads * self.head_dim
        self.kv_dim = self.kv_heads * self.head_dim
        self.qkv_proj = nn.Linear(
            self.embedding_dim,
            self.q_dim + (2 * self.kv_dim),
            bias=config.bias,
        )
        self.out_proj = nn.Linear(self.q_dim, self.embedding_dim, bias=config.bias)

        self.has_qk_norm = config.qk_norm is not None
        if self.has_qk_norm:
            self.q_norm = config.qk_norm.build(self.head_dim)
            self.k_norm = config.qk_norm.build(self.head_dim)

        self.attn_dropout = nn.Dropout(config.attn_dropout)
        self.resid_dropout = nn.Dropout(config.resid_dropout)

        self.has_flash_attn = hasattr(F, "scaled_dot_product_attention")

    def _get_num_kv_heads(self, config: AttentionLayerConfig):
        return config.num_heads

    def get_qkv(self, x: torch.Tensor):
        # x has dimensionality of (batch_size, sequence_length, embedding_dim).
        # (B, T, C) -> (B, T, q_dim + 2*kv_dim)
        return self.qkv_proj(x).split(
            (self.q_dim, self.kv_dim, self.kv_dim),
            dim=-1,
        )

    def get_heads(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        B, T, _ = query.size()  # pylint: disable=invalid-name

        # Queries use every attention head, while keys and values may use fewer
        # shared heads for MQA and GQA.
        q_heads = query.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k_heads = key.view(B, key.size(dim=1), self.kv_heads, self.head_dim).transpose(
            1, 2
        )
        v_heads = value.view(
            B, value.size(dim=1), self.kv_heads, self.head_dim
        ).transpose(1, 2)

        return q_heads, k_heads, v_heads

    def _repeat_kv_heads(self, key: torch.Tensor, value: torch.Tensor):
        if 1 < self.kv_heads < self.num_heads:
            key = key.repeat_interleave(self.shared_heads, dim=1)
            value = value.repeat_interleave(self.shared_heads, dim=1)

        return key, value

    def _build_base_attention_mask(
        self,
        query_length,
        key_length,
        device,
        _attention_positions=None,
    ):
        return torch.ones(
            1,
            1,
            query_length,
            key_length,
            dtype=torch.bool,
            device=device,
        )

    def _build_explicit_attention_mask(
        self,
        attention_mask,
        query,
        key,
        attention_positions=None,
    ):
        # Combine structural and padding constraints into one boolean mask.
        base_mask = self._build_base_attention_mask(
            query.size(2),
            key.size(2),
            query.device,
            attention_positions,
        )
        if attention_mask is None:
            return base_mask

        if (
            attention_mask.ndim != 2
            or attention_mask.size(0) != query.size(0)
            or attention_mask.size(1) < key.size(2)
        ):
            raise ValueError(
                "Attention mask must contain the complete key sequence shape."
            )

        key_padding_mask = attention_mask[:, -key.size(2) :].to(base_mask.device).ne(0)
        combined_mask = base_mask & key_padding_mask[:, None, None, :]

        # Left-padded query rows can have no valid keys. Their outputs are ignored,
        # but keeping a causal row prevents all-masked softmax rows from producing NaNs.
        return torch.where(
            combined_mask.any(dim=-1, keepdim=True),
            combined_mask,
            base_mask,
        )

    def _prepare_attention_mask(
        self,
        attention_mask,
        query,
        key,
        attention_positions=None,
    ):
        # Prefer SDPA's implicit causal mode when no explicit mask is needed.
        is_fully_causal = (
            self.is_causal and self.window_size is None and query.size(2) == key.size(2)
        )
        use_causal_mask = (
            self.has_flash_attn and attention_mask is None and is_fully_causal
        )

        combined_attention_mask = None
        if (
            not self.has_flash_attn
            or attention_mask is not None
            or (self.is_causal and not use_causal_mask)
        ):
            combined_attention_mask = self._build_explicit_attention_mask(
                attention_mask,
                query,
                key,
                attention_positions,
            )
        return combined_attention_mask, use_causal_mask

    def attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask=None,
    ):
        # (B, num_heads, T, head_dim) @ (B, num_heads, head_dim, T) -> (B, num_heads, T, T)
        scale = 1.0 / math.sqrt(key.size(dim=-1))
        attn = (query @ key.transpose(-2, -1)) * scale

        # Apply mask to Q@K^T matrix. Where the mask is equal
        # to zero we will replace the matrix's element at
        # that position with -inf. We use -inf so when we apply
        # softmax those elements will equal to 0.
        if attention_mask is not None:
            attn = attn.masked_fill(~attention_mask, float("-inf"))
        attn = F.softmax(attn, dim=-1)

        attn = self.attn_dropout(attn)

        # (B, num_heads, T, T) x (B, num_heads, T, head_dim) -> (B, num_heads, T, head_dim)
        return attn @ value

    def forward(
        self,
        x,
        attention_mask=None,
        qk_position_data=None,
        kv_cache: KVCache = None,
        cache_position=None,
    ):
        if kv_cache is not None:
            assert (
                not self.training
            ), "KV caching requires attention to be in eval mode."
            assert self.supports_kv_cache, "KV caching requires causal attention."
            if cache_position is None:
                raise ValueError("Cache positions are required when using a KV cache.")

        # batch size, sequence length, embedding dimensionality.
        B, T, _ = x.size()

        # Project Q, K, and V together, then split their output dimensions.
        q, k, v = self.get_qkv(x)

        # next we split the projected embeddings across the number
        # of heads we have, allowing each head to gain a different
        # interpretation.
        # (B, num_head, T, head_dim)
        q, k, v = self.get_heads(q, k, v)
        if self.has_qk_norm:
            q = self.q_norm(q).to(dtype=q.dtype)
            k = self.k_norm(k).to(dtype=k.dtype)

        if self.qk_positional_embedding is not None:
            q, k = self.qk_positional_embedding.apply_qk(q, k, qk_position_data)

        attention_positions = None
        if kv_cache is not None:
            k, v, key_mask, key_positions = kv_cache.update(
                k,
                v,
                cache_position,
                attention_mask,
            )
            attention_mask = key_mask
            attention_positions = (cache_position, key_positions)

        k, v = self._repeat_kv_heads(k, v)

        combined_attention_mask, use_causal_mask = self._prepare_attention_mask(
            attention_mask,
            q,
            k,
            attention_positions,
        )
        if self.has_flash_attn:
            attn = F.scaled_dot_product_attention(  # pylint: disable=not-callable
                q,
                k,
                v,
                attn_mask=combined_attention_mask,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=use_causal_mask,
            )
        else:
            attn = self.attention(
                q,
                k,
                v,
                attention_mask=combined_attention_mask,
            )

        # Convert multi-headed shaped matrix back to original shape
        # (B, num_heads, T, head_dim) -> (B, T, num_heads, head_dim)
        # -> (B*T*num_heads*head_dim) -> (B, T, q_dim)
        attn = attn.transpose(1, 2).contiguous().view(B, T, self.q_dim)

        out = self.out_proj(attn)
        out = self.resid_dropout(out)

        return out
