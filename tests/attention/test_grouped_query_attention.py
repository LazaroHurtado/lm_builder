import torch

from lm_builder.attention import AttentionConfig, GroupedQueryAttention


def test_grouped_query_attention_shares_key_value_heads():
    attention = GroupedQueryAttention(
        AttentionConfig(
            context_length=4,
            embedding_dimension=8,
            num_heads=4,
            kv_heads=2,
        )
    )
    inputs = torch.randn(2, 4, 8)
    query, key, value = attention.get_qkv(inputs)

    query_heads, key_heads, value_heads = attention.get_heads(query, key, value)

    original_key_heads = key.view(2, 4, 2, 2).transpose(1, 2)
    original_value_heads = value.view(2, 4, 2, 2).transpose(1, 2)
    assert query_heads.shape == (2, 4, 4, 2)
    assert torch.equal(
        key_heads,
        original_key_heads.repeat_interleave(2, dim=1),
    )
    assert torch.equal(
        value_heads,
        original_value_heads.repeat_interleave(2, dim=1),
    )

    output = attention(inputs)
    assert output.shape == inputs.shape
