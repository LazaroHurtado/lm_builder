import pytest
import torch
from torch import nn
from torch.nn import functional as F

from lm_builder.attention import (
    AttentionConfig,
    CausalMultiHeadAttention,
    GroupedQueryAttention,
    MultiQueryAttention,
)
from lm_builder.inference import KVCache
from lm_builder.normalizers import NormalizerConfig, RMSNorm


class RecordingPositionalEmbedding(nn.Module):
    def __init__(self, *_):
        super().__init__()
        self.query = None
        self.key = None

    def forward(self, query, key, **_):
        self.query = query.detach().clone()
        self.key = key.detach().clone()
        return query, key


def test_scaled_dot_product_attention_dropout_respects_module_mode(monkeypatch):
    dropout_probabilities = []

    def scaled_dot_product_attention(query, key, value, **kwargs):
        dropout_probabilities.append(kwargs["dropout_p"])
        return torch.zeros_like(query)

    monkeypatch.setattr(
        F,
        "scaled_dot_product_attention",
        scaled_dot_product_attention,
    )
    attention = CausalMultiHeadAttention(
        AttentionConfig(
            context_length=4,
            embedding_dimension=8,
            num_heads=2,
            attention_type=CausalMultiHeadAttention,
            attn_dropout=0.5,
        )
    )
    inputs = torch.randn(1, 4, 8)

    attention.eval()
    attention(inputs)
    attention.train()
    attention(inputs)

    assert dropout_probabilities == [0.0, 0.5]


@pytest.mark.parametrize(
    ("attention_type", "kv_heads", "kv_dimension"),
    [
        (CausalMultiHeadAttention, 4, 8),
        (MultiQueryAttention, 1, 2),
        (GroupedQueryAttention, 2, 4),
    ],
)
def test_attention_uses_one_equivalent_qkv_projection(
    attention_type,
    kv_heads,
    kv_dimension,
):
    torch.manual_seed(17)
    embedding_dimension = 8
    attention = attention_type(
        AttentionConfig(
            context_length=4,
            embedding_dimension=embedding_dimension,
            num_heads=4,
            attention_type=attention_type,
            kv_heads=kv_heads,
            bias=True,
        )
    )
    query_projection = nn.Linear(embedding_dimension, embedding_dimension)
    key_projection = nn.Linear(embedding_dimension, kv_dimension)
    value_projection = nn.Linear(embedding_dimension, kv_dimension)
    with torch.no_grad():
        attention.qkv_proj.weight.copy_(
            torch.cat(
                (
                    query_projection.weight,
                    key_projection.weight,
                    value_projection.weight,
                )
            )
        )
        attention.qkv_proj.bias.copy_(
            torch.cat(
                (
                    query_projection.bias,
                    key_projection.bias,
                    value_projection.bias,
                )
            )
        )

    inputs = torch.randn(2, 4, embedding_dimension)
    query, key, value = attention.get_qkv(inputs)

    assert attention.qkv_proj.out_features == embedding_dimension + 2 * kv_dimension
    assert not hasattr(attention, "q_proj")
    assert not hasattr(attention, "k_proj")
    assert not hasattr(attention, "v_proj")

    # Fused and separate GEMMs can accumulate float32 values differently.
    torch.testing.assert_close(query, query_projection(inputs), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(key, key_projection(inputs), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(value, value_projection(inputs), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    ("attention_type", "kv_heads"),
    [
        (CausalMultiHeadAttention, 4),
        (MultiQueryAttention, 1),
        (GroupedQueryAttention, 2),
    ],
)
def test_attention_builds_independent_per_head_qk_norms(
    attention_type,
    kv_heads,
):
    attention = attention_type(
        AttentionConfig(
            context_length=4,
            embedding_dimension=8,
            num_heads=4,
            attention_type=attention_type,
            kv_heads=kv_heads,
            qk_norm=NormalizerConfig.build_config(
                {
                    "type": "RMSNorm",
                    "eps": 1e-5,
                }
            ),
        )
    )

    output = attention(torch.randn(2, 4, 8))
    output.square().mean().backward()

    assert isinstance(attention.q_norm, RMSNorm)
    assert isinstance(attention.k_norm, RMSNorm)
    assert attention.q_norm is not attention.k_norm
    assert attention.q_norm.weight.shape == (attention.head_dim,)
    assert attention.k_norm.weight.shape == (attention.head_dim,)
    assert attention.q_norm.weight.data_ptr() != attention.k_norm.weight.data_ptr()
    assert attention.q_norm.weight.grad.abs().sum() > 0
    assert attention.k_norm.weight.grad.abs().sum() > 0


def test_qk_norm_runs_before_positional_embeddings_and_kv_repetition():
    attention = GroupedQueryAttention(
        AttentionConfig(
            context_length=4,
            embedding_dimension=8,
            num_heads=4,
            attention_type=GroupedQueryAttention,
            kv_heads=2,
            positional_embedding=RecordingPositionalEmbedding,
            qk_norm=NormalizerConfig.build_config({"type": "RMSNorm"}),
        )
    )
    inputs = torch.randn(2, 4, 8)
    with torch.no_grad():
        query, key, value = attention.get_qkv(inputs)
        query, key, _ = attention.get_heads(query, key, value)
        expected_query = attention.q_norm(query)
        expected_key = attention.k_norm(key)
        attention(inputs)

    assert attention.pos_emb.query.shape == (2, 4, 4, 2)
    assert attention.pos_emb.key.shape == (2, 2, 4, 2)
    torch.testing.assert_close(attention.pos_emb.query, expected_query)
    torch.testing.assert_close(attention.pos_emb.key, expected_key)


def test_qk_norm_adds_no_modules_or_parameters_when_disabled():
    attention = CausalMultiHeadAttention(
        AttentionConfig(
            context_length=4,
            embedding_dimension=8,
            num_heads=4,
            attention_type=CausalMultiHeadAttention,
        )
    )

    assert not attention.has_qk_norm
    assert not hasattr(attention, "q_norm")
    assert not hasattr(attention, "k_norm")
    assert all(
        "q_norm" not in name and "k_norm" not in name for name in attention.state_dict()
    )


def test_qk_norm_preserves_kv_dtype_during_autocast():
    attention = CausalMultiHeadAttention(
        AttentionConfig(
            context_length=4,
            embedding_dimension=8,
            num_heads=4,
            attention_type=CausalMultiHeadAttention,
            qk_norm=NormalizerConfig.build_config({"type": "RMSNorm"}),
        )
    ).eval()
    kv_cache = KVCache(capacity=4)

    with torch.autocast("cpu", dtype=torch.bfloat16):
        attention(torch.randn(2, 4, 8), kv_cache=kv_cache)

    assert kv_cache.k.dtype is torch.bfloat16
    assert kv_cache.v.dtype is torch.bfloat16
