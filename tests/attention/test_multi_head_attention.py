import torch
from torch.nn import functional as F

from lm_builder.attention import AttentionConfig, CausalMultiHeadAttention


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
            attn_dropout=0.5,
        )
    )
    inputs = torch.randn(1, 4, 8)

    attention.eval()
    attention(inputs)
    attention.train()
    attention(inputs)

    assert dropout_probabilities == [0.0, 0.5]
