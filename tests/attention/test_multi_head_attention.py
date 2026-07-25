import pytest
import torch
from torch import nn
from torch.nn import functional as F

from lm_builder.attention import (
    AttentionLayerConfig,
    CausalMultiHeadAttention,
    GroupedQueryAttention,
)
from lm_builder.kv_cache import KVCache
from lm_builder.normalizers import NormalizerConfig, RMSNorm


class RecordingPositionalEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.query = None
        self.key = None

    def prepare(self, *_args):
        return self

    @staticmethod
    def apply_qk(query, key, position_data):
        position_data.query = query.detach().clone()
        position_data.key = key.detach().clone()
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
        AttentionLayerConfig(
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
        (GroupedQueryAttention, 1, 2),
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
        AttentionLayerConfig(
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


def test_attention_infers_head_dimension_from_embedding_dimension():
    attention = CausalMultiHeadAttention(
        AttentionLayerConfig(
            context_length=4,
            embedding_dimension=8,
            num_heads=4,
            attention_type=CausalMultiHeadAttention,
        )
    )

    assert attention.head_dim == 2
    assert attention.q_dim == 8
    assert attention.kv_dim == 8
    assert attention.qkv_proj.out_features == 24
    assert attention.out_proj.in_features == 8
    assert attention.out_proj.out_features == 8
    assert attention(torch.randn(2, 4, 8)).shape == (2, 4, 8)


def test_inferred_head_dimension_uses_embedding_to_head_ratio():
    config = AttentionLayerConfig(
        context_length=4,
        embedding_dimension=10,
        num_heads=4,
        attention_type=CausalMultiHeadAttention,
    )
    attention = CausalMultiHeadAttention(config)

    assert config.head_dim == 2
    assert attention.head_dim == 2
    assert attention(torch.randn(2, 4, 10)).shape == (2, 4, 10)


def test_explicit_head_dimension_sizes_fused_and_output_projections():
    attention = GroupedQueryAttention(
        AttentionLayerConfig(
            context_length=4,
            embedding_dimension=8,
            num_heads=4,
            head_dim=3,
            attention_type=GroupedQueryAttention,
            kv_heads=2,
        )
    )
    inputs = torch.randn(2, 4, 8)

    query, key, value = attention.get_qkv(inputs)

    assert attention.head_dim == 3
    assert attention.q_dim == 12
    assert attention.kv_dim == 6
    assert attention.qkv_proj.in_features == 8
    assert attention.qkv_proj.out_features == 24
    assert query.shape == (2, 4, 12)
    assert key.shape == (2, 4, 6)
    assert value.shape == (2, 4, 6)
    assert attention.out_proj.in_features == 12
    assert attention.out_proj.out_features == 8
    assert attention(inputs).shape == (2, 4, 8)


@pytest.mark.parametrize(
    ("attention_type", "kv_heads"),
    [
        (CausalMultiHeadAttention, 4),
        (GroupedQueryAttention, 1),
        (GroupedQueryAttention, 2),
    ],
)
def test_attention_builds_independent_per_head_qk_norms(
    attention_type,
    kv_heads,
):
    attention = attention_type(
        AttentionLayerConfig(
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
    positional_embedding = RecordingPositionalEmbedding()
    attention = GroupedQueryAttention(
        AttentionLayerConfig(
            context_length=4,
            embedding_dimension=8,
            num_heads=4,
            attention_type=GroupedQueryAttention,
            kv_heads=2,
            qk_norm=NormalizerConfig.build_config({"type": "RMSNorm"}),
        ),
        qk_positional_embedding=positional_embedding,
    )
    inputs = torch.randn(2, 4, 8)
    with torch.no_grad():
        query, key, value = attention.get_qkv(inputs)
        query, key, _ = attention.get_heads(query, key, value)
        expected_query = attention.q_norm(query)
        expected_key = attention.k_norm(key)
        attention(
            inputs,
            qk_position_data=positional_embedding,
        )

    assert positional_embedding.query.shape == (2, 4, 4, 2)
    assert positional_embedding.key.shape == (2, 2, 4, 2)
    torch.testing.assert_close(positional_embedding.query, expected_query)
    torch.testing.assert_close(positional_embedding.key, expected_key)


def test_qk_norm_adds_no_modules_or_parameters_when_disabled():
    attention = CausalMultiHeadAttention(
        AttentionLayerConfig(
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
        AttentionLayerConfig(
            context_length=4,
            embedding_dimension=8,
            num_heads=4,
            attention_type=CausalMultiHeadAttention,
            qk_norm=NormalizerConfig.build_config({"type": "RMSNorm"}),
        )
    ).eval()
    kv_cache = KVCache(capacity=4)

    with torch.autocast("cpu", dtype=torch.bfloat16):
        attention(
            torch.randn(2, 4, 8),
            kv_cache=kv_cache,
            cache_position=torch.arange(4),
        )

    assert kv_cache.k.dtype is torch.bfloat16
    assert kv_cache.v.dtype is torch.bfloat16


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS is required for this Inductor regression test.",
)
def test_compiled_mps_decode_preserves_large_indexed_cache():
    attention = CausalMultiHeadAttention(
        AttentionLayerConfig(
            context_length=4109,
            embedding_dimension=8,
            num_heads=2,
            attention_type=CausalMultiHeadAttention,
        )
    ).eval()
    attention.to("mps")
    kv_cache = KVCache(capacity=4109)
    attention(
        torch.randn(1, 13, 8, device="mps"),
        attention_mask=torch.ones(1, 13, dtype=torch.bool, device="mps"),
        kv_cache=kv_cache,
        cache_position=torch.arange(13, device="mps"),
    )
    retained_key = kv_cache.k[:, :, :13].clone()
    retained_value = kv_cache.v[:, :, :13].clone()
    compiled_attention = torch.compile(attention, fullgraph=True)

    output = compiled_attention(
        torch.randn(1, 1, 8, device="mps"),
        attention_mask=torch.ones(1, 1, dtype=torch.bool, device="mps"),
        kv_cache=kv_cache,
        cache_position=torch.tensor([13], device="mps"),
    )
    torch.mps.synchronize()

    assert output.shape == (1, 1, 8)
    assert torch.equal(kv_cache.k[:, :, :13], retained_key)
    assert torch.equal(kv_cache.v[:, :, :13], retained_value)
