import torch

from lm_builder.attention import AttentionLayerConfig, GroupedQueryAttention
from lm_builder.kv_cache import KVCache


def test_grouped_query_attention_shares_key_value_heads():
    attention = GroupedQueryAttention(
        AttentionLayerConfig(
            context_length=4,
            embedding_dimension=8,
            num_heads=4,
            attention_type=GroupedQueryAttention,
            kv_heads=2,
        )
    )
    inputs = torch.randn(2, 4, 8)
    query, key, value = attention.get_qkv(inputs)

    query_heads, key_heads, value_heads = attention.get_heads(query, key, value)

    original_key_heads = key.view(2, 4, 2, 2).transpose(1, 2)
    original_value_heads = value.view(2, 4, 2, 2).transpose(1, 2)
    assert query_heads.shape == (2, 4, 4, 2)
    assert torch.equal(key_heads, original_key_heads)
    assert torch.equal(value_heads, original_value_heads)

    repeated_key_heads, repeated_value_heads = attention._repeat_kv_heads(
        key_heads,
        value_heads,
    )
    assert torch.equal(
        repeated_key_heads,
        original_key_heads.repeat_interleave(2, dim=1),
    )
    assert torch.equal(
        repeated_value_heads,
        original_value_heads.repeat_interleave(2, dim=1),
    )

    output = attention(inputs)
    assert output.shape == inputs.shape


def test_grouped_query_attention_and_kv_cache_use_explicit_head_dimension():
    attention = GroupedQueryAttention(
        AttentionLayerConfig(
            context_length=4,
            embedding_dimension=8,
            num_heads=4,
            head_dim=3,
            attention_type=GroupedQueryAttention,
            kv_heads=2,
        )
    ).eval()
    inputs = torch.randn(2, 4, 8)
    query, key, value = attention.get_qkv(inputs)

    query_heads, key_heads, value_heads = attention.get_heads(query, key, value)
    repeated_key_heads, repeated_value_heads = attention._repeat_kv_heads(
        key_heads,
        value_heads,
    )
    cache = KVCache(capacity=4)
    output = attention(
        inputs,
        kv_cache=cache,
        cache_position=torch.arange(inputs.size(1)),
    )

    assert query_heads.shape == (2, 4, 4, 3)
    assert key_heads.shape == (2, 2, 4, 3)
    assert value_heads.shape == (2, 2, 4, 3)
    assert repeated_key_heads.shape == (2, 4, 4, 3)
    assert repeated_value_heads.shape == (2, 4, 4, 3)
    assert cache.k.shape == (2, 2, 4, 3)
    assert cache.v.shape == (2, 2, 4, 3)
    assert output.shape == (2, 4, 8)
