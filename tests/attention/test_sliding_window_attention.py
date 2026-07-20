import pytest
import torch

from lm_builder.attention import AttentionConfig, SlidingWindowAttention
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


def test_transformer_config_resolves_sliding_window_attention():
    config = TransformerConfig.build_config(
        {
            "attention": "SlidingWindowAttention",
            "attention_config": {
                "context_length": 4,
                "embedding_dimension": 8,
                "num_heads": 2,
                "window_size": 2,
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

    assert isinstance(model.transformer.blocks[0].attn, SlidingWindowAttention)
    logits, loss = model(torch.tensor([[1, 2, 3, 4]]))
    assert logits.shape == (1, 4, 10)
    assert loss is None
