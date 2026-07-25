import pytest
import torch
from torch.nn import functional as F

from lm_builder.attention import (
    AttentionLayerConfig,
    CausalMultiHeadAttention,
    GroupedQueryAttention,
    MultiHeadAttention,
    MultiQueryAttention,
)


@pytest.mark.parametrize(
    "attention_type",
    [
        CausalMultiHeadAttention,
        MultiQueryAttention,
        GroupedQueryAttention,
    ],
)
def test_causal_attention_builds_windowed_mask(attention_type):
    attention = attention_type(
        AttentionLayerConfig(
            context_length=5,
            embedding_dimension=8,
            num_heads=4,
            attention_type=attention_type,
            kv_heads=2,
            window_size=3,
        )
    )

    attention_mask = attention._build_base_attention_mask(
        query_length=5,
        key_length=5,
        device=attention.qkv_proj.weight.device,
    )

    expected_mask = torch.tensor(
        [
            [1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 1, 1],
        ],
        dtype=torch.bool,
    )[None, None, :, :]
    assert torch.equal(attention_mask, expected_mask)


def test_windowed_mask_aligns_cached_queries_to_the_end():
    attention = GroupedQueryAttention(
        AttentionLayerConfig(
            context_length=5,
            embedding_dimension=8,
            num_heads=4,
            attention_type=GroupedQueryAttention,
            kv_heads=2,
            window_size=3,
        )
    )

    attention_mask = attention._build_base_attention_mask(
        query_length=1,
        key_length=5,
        device=attention.qkv_proj.weight.device,
    )

    expected_mask = torch.tensor(
        [[0, 0, 1, 1, 1]],
        dtype=torch.bool,
    )[None, None, :, :]
    assert torch.equal(attention_mask, expected_mask)


def test_windowed_mask_uses_logical_ring_buffer_positions():
    attention = GroupedQueryAttention(
        AttentionLayerConfig(
            context_length=5,
            embedding_dimension=8,
            num_heads=4,
            attention_type=GroupedQueryAttention,
            kv_heads=2,
            window_size=3,
        )
    )

    attention_mask = attention._build_base_attention_mask(
        query_length=1,
        key_length=5,
        device=attention.qkv_proj.weight.device,
        attention_positions=(
            torch.tensor([5]),
            torch.tensor([5, 1, 2, 3, 4]),
        ),
    )

    expected_mask = torch.tensor(
        [[1, 0, 0, 1, 1]],
        dtype=torch.bool,
    )[None, None, :, :]
    assert torch.equal(attention_mask, expected_mask)


def test_causal_attention_without_window_builds_full_triangle():
    attention = GroupedQueryAttention(
        AttentionLayerConfig(
            context_length=4,
            embedding_dimension=8,
            num_heads=4,
            attention_type=GroupedQueryAttention,
            kv_heads=2,
        )
    )

    attention_mask = attention._build_base_attention_mask(
        query_length=4,
        key_length=4,
        device=attention.qkv_proj.weight.device,
    )

    expected_mask = torch.ones(4, 4, dtype=torch.bool).tril()[None, None, :, :]
    assert torch.equal(attention_mask, expected_mask)


def test_windowed_attention_uses_only_the_current_window():
    attention = CausalMultiHeadAttention(
        AttentionLayerConfig(
            context_length=4,
            embedding_dimension=1,
            num_heads=1,
            attention_type=CausalMultiHeadAttention,
            window_size=2,
        )
    )
    query = torch.zeros(1, 1, 4, 1)
    key = torch.zeros(1, 1, 4, 1)
    value = torch.tensor([[[[1.0], [2.0], [4.0], [8.0]]]])
    attention_mask = attention._build_base_attention_mask(
        query_length=4,
        key_length=4,
        device=attention.qkv_proj.weight.device,
    )

    output = F.scaled_dot_product_attention(  # pylint: disable=not-callable
        query,
        key,
        value,
        attn_mask=attention_mask,
    )

    expected_output = torch.tensor([[[[1.0], [1.5], [3.0], [6.0]]]])
    assert torch.allclose(output, expected_output)


@pytest.mark.parametrize("window_size", [0, -1, 1.5, True, "3"])
def test_attention_config_rejects_invalid_window_size(window_size):
    with pytest.raises(
        ValueError,
        match="window_size must be a positive integer or None",
    ):
        AttentionLayerConfig(
            context_length=4,
            embedding_dimension=8,
            num_heads=2,
            attention_type=CausalMultiHeadAttention,
            window_size=window_size,
        )


def test_attention_does_not_register_context_sized_mask():
    attention = CausalMultiHeadAttention(
        AttentionLayerConfig(
            context_length=32_768,
            embedding_dimension=8,
            num_heads=2,
            attention_type=CausalMultiHeadAttention,
        )
    )

    assert "attention_mask" not in dict(attention.named_buffers())


def test_noncausal_attention_cannot_enable_window_support_directly():
    class NonCausalWindowAttention(MultiHeadAttention):
        supports_window_size = True

    with pytest.raises(
        ValueError,
        match="NonCausalWindowAttention does not support window_size",
    ):
        NonCausalWindowAttention(
            AttentionLayerConfig(
                context_length=4,
                embedding_dimension=8,
                num_heads=2,
                attention_type=NonCausalWindowAttention,
                window_size=2,
            )
        )
