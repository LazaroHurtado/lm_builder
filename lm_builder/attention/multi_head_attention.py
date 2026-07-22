from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from ..inference import KVCache
from .attention import Attention
from .config import AttentionConfig


class MultiHeadAttention(Attention):
    supports_kv_cache = False
    supports_window_size = False
    is_causal = False

    def __init__(self, config: AttentionConfig):
        super().__init__()
        if config.window_size is not None and (
            not self.supports_window_size or not self.is_causal
        ):
            raise ValueError(f"{type(self).__name__} does not support window_size.")

        assert config.embedding_dimension % config.num_heads == 0

        self.context_len = config.context_length
        self.embedding_dim = config.embedding_dimension
        self.num_heads = config.num_heads
        self.window_size = config.window_size

        self.head_dim = self.embedding_dim // self.num_heads

        self.q_proj = nn.Linear(
            self.embedding_dim, self.embedding_dim, bias=config.bias
        )
        self.k_proj = nn.Linear(
            self.embedding_dim, self.embedding_dim, bias=config.bias
        )
        self.v_proj = nn.Linear(
            self.embedding_dim, self.embedding_dim, bias=config.bias
        )
        self.out_proj = nn.Linear(
            self.embedding_dim, self.embedding_dim, bias=config.bias
        )

        self.attn_dropout = nn.Dropout(config.attn_dropout)
        self.resid_dropout = nn.Dropout(config.resid_dropout)

        self.has_positional_embedding = config.positional_embedding is not None
        if self.has_positional_embedding:
            self.pos_emb = config.positional_embedding(
                self.head_dim, config.context_length, config.inv_freq
            )

        self.has_flash_attn = hasattr(F, "scaled_dot_product_attention")

    def get_qkv(self, x: torch.Tensor):
        # x has dimensionality of (batch_size, sequence_length, embedding_dim).
        # (B, T, C) -> (B, T, C)
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        return q, k, v

    def get_heads(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        B, T, _ = query.size()  # pylint: disable=invalid-name

        # (B, T, C) -> (B, T, num_head, head_dim) -> (B, num_head, T, head_dim)
        q_heads = query.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k_heads = key.view(B, key.size(dim=1), self.num_heads, self.head_dim).transpose(
            1, 2
        )
        v_heads = value.view(
            B, value.size(dim=1), self.num_heads, self.head_dim
        ).transpose(1, 2)

        return q_heads, k_heads, v_heads

    def _repeat_kv_heads(self, key: torch.Tensor, value: torch.Tensor):
        return key, value

    def _build_base_attention_mask(
        self,
        query_length,
        key_length,
        device,
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
        batch_size,
        query_length,
        key_length,
    ):
        # Combine structural and padding constraints into one boolean mask.
        base_mask = self._build_base_attention_mask(
            query_length,
            key_length,
            self.q_proj.weight.device,
        )
        if attention_mask is None:
            return base_mask

        if (
            attention_mask.ndim != 2
            or attention_mask.size(0) != batch_size
            or attention_mask.size(1) < key_length
        ):
            raise ValueError(
                "Attention mask must contain the complete key sequence shape."
            )

        key_padding_mask = attention_mask[:, -key_length:].to(
            device=base_mask.device,
            dtype=torch.bool,
        )[:, None, None, :]
        combined_mask = base_mask & key_padding_mask

        # Left-padded query rows can have no valid keys. Their outputs are ignored,
        # but keeping a causal row prevents all-masked softmax rows from producing NaNs.
        return torch.where(
            combined_mask.any(dim=-1, keepdim=True),
            combined_mask,
            base_mask,
        )

    def _prepare_attention_mask(
        self, attention_mask, key_length, query_length, batch_size
    ):
        # Prefer SDPA's implicit causal mode when no explicit mask is needed.
        is_fully_causal = (
            self.is_causal and self.window_size is None and query_length == key_length
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
                batch_size,
                query_length,
                key_length,
            )
        return combined_attention_mask, use_causal_mask

    def attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask,
    ):
        # (B, num_heads, T, head_dim) @ (B, num_heads, head_dim, T) -> (B, num_heads, T, T)
        scale = 1.0 / math.sqrt(key.size(dim=-1))
        attn = (query @ key.transpose(-2, -1)) * scale

        # Apply mask to Q@K^T matrix. Where the mask is equal
        # to zero we will replace the matrix's element at
        # that position with -inf. We use -inf so when we apply
        # softmax those elements will equal to 0.
        attn = attn.masked_fill(~attention_mask, float("-inf"))
        attn = F.softmax(attn, dim=-1)

        attn = self.attn_dropout(attn)

        # (B, num_heads, T, T) x (B, num_heads, T, head_dim) -> (B, num_heads, T, head_dim)
        return attn @ value

    def forward(
        self,
        x,
        attention_mask=None,
        position_ids=None,
        kv_cache: KVCache = None,
    ):
        if kv_cache is not None:
            assert (
                not self.training
            ), "KV caching requires attention to be in eval mode."
            assert self.supports_kv_cache, "KV caching requires causal attention."

        # batch size, sequence length, embedding dimensionality.
        B, T, C = x.size()

        # we get the q, k, v projection of each embedding, each
        # matrix will have dimension (B, T, C)
        q, k, v = self.get_qkv(x)

        # next we split the projected embeddings across the number
        # of heads we have, allowing each head to gain a different
        # interpretation.
        # (B, num_head, T, head_dim)
        q, k, v = self.get_heads(q, k, v)
        if self.has_positional_embedding:
            q, k = self.pos_emb(q, k, position_ids=position_ids)

        if kv_cache is not None:
            k, v = kv_cache.update(k, v)

        k, v = self._repeat_kv_heads(k, v)

        combined_attention_mask, use_causal_mask = self._prepare_attention_mask(
            attention_mask, key_length=k.size(2), query_length=T, batch_size=B
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
        # -> (B*T*num_heads*head_dim) -> (B, T, C)
        attn = attn.transpose(1, 2).contiguous().view(B, T, C)

        out = self.out_proj(attn)
        out = self.resid_dropout(out)

        return out
