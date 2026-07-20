import pytest
import torch
from torch.nn import functional as F

from lm_builder.attention import AttentionConfig, CausalMultiHeadAttention
from lm_builder.ffn import FeedForward, FeedForwardConfig
from lm_builder.positional_embeddings import AbsolutePE, RotaryPE
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


@pytest.mark.parametrize("position_type", ["absolute", "rotary"])
@pytest.mark.parametrize("use_scaled_dot_product_attention", [True, False])
def test_left_padding_does_not_change_content_logits(
    position_type,
    use_scaled_dot_product_attention,
):
    attention_config = AttentionConfig(
        context_length=4,
        embedding_dimension=8,
        num_heads=2,
        positional_embedding=RotaryPE if position_type == "rotary" else None,
    )
    model = Transformer(
        TransformerConfig(
            attention_config=attention_config,
            ffn_config=FeedForwardConfig(
                embedding_dimension=8,
                intermediate_dimension=16,
            ),
            vocab_size=10,
            num_layers=1,
            attention=CausalMultiHeadAttention,
            ffn=FeedForward,
            positional_embedding=AbsolutePE if position_type == "absolute" else None,
        )
    ).eval()
    model.transformer.blocks[0].attn.has_flash_attn = (
        use_scaled_dot_product_attention and hasattr(F, "scaled_dot_product_attention")
    )
    input_ids = torch.tensor([[2, 3]])
    attention_mask = torch.tensor([[1, 1]])
    padded_input_ids = torch.tensor([[0, 0, 2, 3], [4, 5, 6, 7]])
    padded_attention_mask = torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]])

    with torch.no_grad():
        logits, _ = model(input_ids, attention_mask=attention_mask)
        padded_logits, _ = model(
            padded_input_ids,
            attention_mask=padded_attention_mask,
        )

    assert torch.allclose(
        logits,
        padded_logits[:1, -input_ids.size(1) :],
        atol=1e-6,
    )
