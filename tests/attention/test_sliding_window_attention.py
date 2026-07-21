import pytest
import torch

from lm_builder.attention import (
    AttentionConfig,
    CausalMultiHeadAttention,
    GroupedQueryAttention,
    SlidingWindowAttention,
)
from lm_builder.ffn import FeedForward, FeedForwardConfig
from lm_builder.transformer import Transformer, TransformerConfig


def test_sliding_window_attention_registers_banded_causal_mask():
    attention = SlidingWindowAttention(
        AttentionConfig(
            context_length=5,
            embedding_dimension=8,
            num_heads=2,
            window_size=3,
        )
    )

    expected_mask = torch.tensor(
        [
            [1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 1, 1],
        ],
        dtype=torch.float,
    )[None, None, :, :]

    assert torch.equal(attention.attention_mask, expected_mask)


def test_sliding_window_attention_uses_only_the_current_window():
    attention = SlidingWindowAttention(
        AttentionConfig(
            context_length=4,
            embedding_dimension=1,
            num_heads=1,
            window_size=2,
        )
    )
    query = torch.zeros(1, 1, 4, 1)
    key = torch.zeros(1, 1, 4, 1)
    value = torch.tensor([[[[1.0], [2.0], [4.0], [8.0]]]])

    output = attention.attention(query, key, value)

    expected_output = torch.tensor([[[[1.0], [1.5], [3.0], [6.0]]]])
    assert torch.allclose(output, expected_output)


@pytest.mark.parametrize("window_size", [None, 0, -1, 1.5, True])
def test_sliding_window_attention_requires_positive_integer_window(window_size):
    with pytest.raises(ValueError, match="window_size must be a positive integer"):
        SlidingWindowAttention(
            AttentionConfig(
                context_length=4,
                embedding_dimension=8,
                num_heads=2,
                window_size=window_size,
            )
        )


@pytest.mark.parametrize(
    ("attention_ratio", "expected"),
    [
        ("1:1", (1, 1)),
        ("5:2", (5, 2)),
        ("5:2:3", (5, 2, 3)),
        ("12:3", (12, 3)),
    ],
)
def test_attention_config_parses_attention_ratio(attention_ratio, expected):
    config = AttentionConfig(
        context_length=4,
        embedding_dimension=8,
        num_heads=2,
        attention_ratio=attention_ratio,
    )

    assert config.get_attention_ratio() == expected


@pytest.mark.parametrize(
    "attention_ratio",
    [1, "", "5", "5:", ":2", "0:1", "1:0", "-1:2", " 5:2", "5:2 "],
)
def test_attention_config_rejects_invalid_attention_ratio(attention_ratio):
    with pytest.raises(
        ValueError,
        match=(
            "attention_ratio must contain at least two colon-separated "
            "positive integers"
        ),
    ):
        AttentionConfig(
            context_length=4,
            embedding_dimension=8,
            num_heads=2,
            attention_ratio=attention_ratio,
        )


def test_transformer_repeats_arbitrary_attention_ratio():
    model = Transformer(
        TransformerConfig(
            attention_config=AttentionConfig(
                context_length=4,
                embedding_dimension=8,
                num_heads=4,
                kv_heads=2,
                window_size=2,
                attention_ratio="5:2:3",
            ),
            ffn_config=FeedForwardConfig(
                embedding_dimension=8,
                intermediate_dimension=16,
            ),
            vocab_size=10,
            num_layers=12,
            attention=[
                SlidingWindowAttention,
                GroupedQueryAttention,
                CausalMultiHeadAttention,
            ],
            ffn=FeedForward,
        )
    )

    attention_types = [type(block.attn) for block in model.transformer.blocks]
    assert attention_types == [
        SlidingWindowAttention,
        SlidingWindowAttention,
        SlidingWindowAttention,
        SlidingWindowAttention,
        SlidingWindowAttention,
        GroupedQueryAttention,
        GroupedQueryAttention,
        CausalMultiHeadAttention,
        CausalMultiHeadAttention,
        CausalMultiHeadAttention,
        SlidingWindowAttention,
        SlidingWindowAttention,
    ]


def test_attention_ratio_requires_attention_list():
    with pytest.raises(
        ValueError,
        match="attention must be a list when attention_ratio is set",
    ):
        TransformerConfig(
            attention_config=AttentionConfig(
                context_length=4,
                embedding_dimension=8,
                num_heads=2,
                window_size=2,
                attention_ratio="1:1",
            ),
            ffn_config=FeedForwardConfig(
                embedding_dimension=8,
                intermediate_dimension=16,
            ),
            vocab_size=10,
            num_layers=2,
            attention=SlidingWindowAttention,
            ffn=FeedForward,
        )


def test_attention_list_requires_attention_ratio():
    with pytest.raises(
        ValueError,
        match="attention can only be a list when attention_ratio is set",
    ):
        TransformerConfig(
            attention_config=AttentionConfig(
                context_length=4,
                embedding_dimension=8,
                num_heads=2,
                window_size=2,
            ),
            ffn_config=FeedForwardConfig(
                embedding_dimension=8,
                intermediate_dimension=16,
            ),
            vocab_size=10,
            num_layers=2,
            attention=[SlidingWindowAttention, CausalMultiHeadAttention],
            ffn=FeedForward,
        )


@pytest.mark.parametrize(
    "attention_types",
    [
        [SlidingWindowAttention],
        [
            SlidingWindowAttention,
            CausalMultiHeadAttention,
            CausalMultiHeadAttention,
        ],
    ],
)
def test_attention_list_length_must_match_ratio(attention_types):
    with pytest.raises(
        ValueError,
        match="attention list length must match attention_ratio component count",
    ):
        TransformerConfig(
            attention_config=AttentionConfig(
                context_length=4,
                embedding_dimension=8,
                num_heads=2,
                window_size=2,
                attention_ratio="1:1",
            ),
            ffn_config=FeedForwardConfig(
                embedding_dimension=8,
                intermediate_dimension=16,
            ),
            vocab_size=10,
            num_layers=2,
            attention=attention_types,
            ffn=FeedForward,
        )


def test_transformer_config_resolves_attention_list():
    config = TransformerConfig.build_config(
        {
            "attention": [
                "SlidingWindowAttention",
                "GroupedQueryAttention",
                "CausalMultiHeadAttention",
            ],
            "attention_config": {
                "context_length": 4,
                "embedding_dimension": 8,
                "num_heads": 4,
                "kv_heads": 2,
                "window_size": 2,
                "attention_ratio": "1:1:1",
            },
            "ffn": "FeedForward",
            "ffn_config": {
                "embedding_dimension": 8,
                "intermediate_dimension": 16,
            },
            "vocab_size": 10,
            "num_layers": 3,
        }
    )

    model = Transformer(config)

    assert type(model.transformer.blocks[0].attn) is SlidingWindowAttention
    assert type(model.transformer.blocks[1].attn) is GroupedQueryAttention
    assert type(model.transformer.blocks[2].attn) is CausalMultiHeadAttention
    logits, loss = model(torch.tensor([[1, 2, 3, 4]]))
    assert logits.shape == (1, 4, 10)
    assert loss is None


def test_transformer_config_resolves_single_attention_without_ratio():
    config = TransformerConfig.build_config(
        {
            "attention": "CausalMultiHeadAttention",
            "attention_config": {
                "context_length": 4,
                "embedding_dimension": 8,
                "num_heads": 2,
            },
            "ffn": "FeedForward",
            "ffn_config": {
                "embedding_dimension": 8,
                "intermediate_dimension": 16,
            },
            "vocab_size": 10,
            "num_layers": 1,
        }
    )

    model = Transformer(config)

    assert type(model.transformer.blocks[0].attn) is CausalMultiHeadAttention
