import torch
from torch.nn import functional as F

from lm_builder.attention import AttentionConfig
from lm_builder.ffn import FeedForwardConfig
from lm_builder.transformer import Transformer, TransformerConfig


def test_forward_computes_cross_entropy_per_token():
    model = Transformer(
        TransformerConfig(
            attention_config=AttentionConfig(
                context_length=4,
                embedding_dimension=8,
                num_heads=2,
            ),
            ffn_config=FeedForwardConfig(
                embedding_dimension=8,
                intermediate_dimension=16,
            ),
            vocab_size=10,
            num_layers=0,
        )
    )
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    targets = torch.tensor([[2, 3, -1], [5, 6, 7]])

    logits, loss = model(input_ids, targets)

    expected_loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=-1,
    )
    assert loss is not None
    assert loss.ndim == 0
    assert torch.allclose(loss, expected_loss)

    loss.backward()
    assert model.lm_head.weight.grad is not None
