import pytest
import torch
from torch.nn import functional as F

from lm_builder.attention import (
    AttentionConfig,
    CausalMultiHeadAttention,
)
from lm_builder.ffn import FeedForward, FeedForwardConfig
from lm_builder.normalizers import RMSNorm
from lm_builder.positional_embeddings import AbsolutePE, RotaryPE
from lm_builder.transformer import Transformer, TransformerConfig


def build_attention_configs(num_layers=1, **kwargs):
    return [
        AttentionConfig(
            attention_type=CausalMultiHeadAttention,
            **kwargs,
        )
        for _ in range(num_layers)
    ]


def build_transformer(tie_word_embeddings=False):
    return Transformer(
        TransformerConfig(
            embedding_dimension=8,
            context_length=4,
            attention_config=build_attention_configs(
                context_length=4,
                embedding_dimension=8,
                num_heads=2,
            ),
            ffn_config=FeedForwardConfig(
                embedding_dimension=8,
                intermediate_dimension=16,
                ffn_type=FeedForward,
            ),
            vocab_size=10,
            num_layers=1,
            tie_word_embeddings=tie_word_embeddings,
        )
    )


def test_forward_computes_cross_entropy_per_token():
    model = Transformer(
        TransformerConfig(
            embedding_dimension=8,
            context_length=4,
            attention_config=build_attention_configs(
                context_length=4,
                embedding_dimension=8,
                num_heads=2,
            ),
            ffn_config=FeedForwardConfig(
                embedding_dimension=8,
                intermediate_dimension=16,
                ffn_type=FeedForward,
            ),
            vocab_size=10,
            num_layers=1,
        )
    )
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    targets = torch.tensor([[2, 3, -1], [5, 6, 7]])

    logits, loss, routing_loss = model(input_ids, targets)

    expected_loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=-1,
    )
    assert loss is not None
    assert loss.ndim == 0
    assert torch.allclose(loss, expected_loss)
    assert routing_loss is None

    loss.backward()
    assert model.lm_head.weight.grad is not None


def test_transformer_builds_self_contained_branch_configs():
    config = TransformerConfig.build_config(
        {
            "embedding_dimension": 8,
            "context_length": 4,
            "attention_config": {
                "context_length": 2,
                "embedding_dimension": 4,
                "num_heads": 2,
                "norm": {
                    "type": "LayerNorm",
                    "eps": 1e-4,
                    "bias": False,
                },
                "layers": [
                    {
                        "type": "CausalMultiHeadAttention",
                        "context_length": 1,
                        "embedding_dimension": 2,
                    }
                ],
            },
            "ffn_config": {
                "type": "FeedForward",
                "embedding_dimension": 16,
                "intermediate_dimension": 16,
                "norm": {
                    "type": "RMSNorm",
                    "eps": 2e-5,
                },
            },
            "norm": {
                "type": "LayerNorm",
                "eps": 3e-5,
                "bias": False,
            },
            "vocab_size": 10,
            "num_layers": 1,
            "tie_word_embeddings": True,
        }
    )

    model = Transformer(config)
    block = model.transformer.blocks[0]

    assert isinstance(block.attn, CausalMultiHeadAttention)
    assert isinstance(block.ffn, FeedForward)
    assert config.attention_config[0].context_length == config.context_length
    assert config.attention_config[0].embedding_dimension == config.embedding_dimension
    assert config.ffn_config.embedding_dimension == config.embedding_dimension
    assert isinstance(block.attn_norm, torch.nn.LayerNorm)
    assert block.attn_norm.eps == 1e-4
    assert isinstance(block.ffn_norm, RMSNorm)
    assert block.ffn_norm.eps == 2e-5
    assert isinstance(model.transformer.norm, torch.nn.LayerNorm)
    assert model.transformer.norm.eps == 3e-5
    assert config.tie_word_embeddings
    assert model.lm_head.weight is model.transformer.wte.weight


def test_tied_word_embeddings_share_parameters_gradients_and_state():
    model = build_transformer(tie_word_embeddings=True)
    shared_weight = model.transformer.wte.weight
    _, loss, _ = model(
        torch.tensor([[1, 2, 3]]),
        targets=torch.tensor([[2, 3, 4]]),
    )

    loss.backward()
    state_dict = model.state_dict()

    assert model.lm_head.weight is shared_weight
    assert model.lm_head.weight.grad is shared_weight.grad
    assert shared_weight.grad.abs().sum() > 0
    assert sum(parameter is shared_weight for parameter in model.parameters()) == 1
    assert (
        state_dict["transformer.wte.weight"].data_ptr()
        == state_dict["lm_head.weight"].data_ptr()
    )


def test_word_embeddings_are_untied_by_default():
    model = build_transformer()

    assert not model.config.tie_word_embeddings
    assert model.lm_head.weight is not model.transformer.wte.weight
    assert model.lm_head.weight.data_ptr() != model.transformer.wte.weight.data_ptr()


def test_tied_word_embeddings_remain_tied_after_assigned_state_load():
    source = build_transformer(tie_word_embeddings=True)
    with torch.no_grad():
        source.transformer.wte.weight.fill_(2.5)

    model = build_transformer(tie_word_embeddings=True).to("meta")
    model.load_state_dict(source.state_dict(), assign=True)

    assert model.lm_head.weight is model.transformer.wte.weight
    assert torch.equal(
        model.transformer.wte.weight,
        torch.full_like(model.transformer.wte.weight, 2.5),
    )


def test_feed_forward_config_requires_type():
    with pytest.raises(ValueError, match="ffn_config.type is required"):
        TransformerConfig(
            embedding_dimension=8,
            context_length=4,
            attention_config=build_attention_configs(
                context_length=4,
                embedding_dimension=8,
                num_heads=2,
            ),
            ffn_config=FeedForwardConfig(
                embedding_dimension=8,
                intermediate_dimension=16,
            ),
            vocab_size=10,
            num_layers=1,
        )


def test_transformer_requires_one_attention_config_per_layer():
    with pytest.raises(
        ValueError,
        match="one AttentionConfig per layer",
    ):
        TransformerConfig(
            embedding_dimension=8,
            context_length=4,
            attention_config=build_attention_configs(
                context_length=4,
                embedding_dimension=8,
                num_heads=2,
            ),
            ffn_config=FeedForwardConfig(
                embedding_dimension=8,
                intermediate_dimension=16,
                ffn_type=FeedForward,
            ),
            vocab_size=10,
            num_layers=2,
        )


@pytest.mark.parametrize(
    ("config_path", "expected_layers", "expected_tied_embeddings"),
    [
        ("examples/gpt2_xl.yml", 48, True),
        ("examples/llama2_7b_chat.yml", 32, False),
        ("examples/tinystories_200m.yml", 17, False),
    ],
)
def test_example_configs_use_resolved_module_configs(
    config_path,
    expected_layers,
    expected_tied_embeddings,
):
    config = TransformerConfig.from_yml(config_path)

    assert len(config.attention_config) == expected_layers
    assert all(layer.attention_type is not None for layer in config.attention_config)
    assert all(
        layer.context_length == config.context_length
        for layer in config.attention_config
    )
    assert all(
        layer.embedding_dimension == config.embedding_dimension
        for layer in config.attention_config
    )
    assert config.ffn_config.embedding_dimension == config.embedding_dimension
    assert config.ffn_config.ffn_type is not None
    assert config.tie_word_embeddings is expected_tied_embeddings


@pytest.mark.parametrize("position_type", ["absolute", "rotary"])
@pytest.mark.parametrize("use_scaled_dot_product_attention", [True, False])
def test_left_padding_does_not_change_content_logits(
    position_type,
    use_scaled_dot_product_attention,
):
    attention_config = build_attention_configs(
        context_length=4,
        embedding_dimension=8,
        num_heads=2,
        positional_embedding=RotaryPE if position_type == "rotary" else None,
    )
    model = Transformer(
        TransformerConfig(
            embedding_dimension=8,
            context_length=4,
            attention_config=attention_config,
            ffn_config=FeedForwardConfig(
                embedding_dimension=8,
                intermediate_dimension=16,
                ffn_type=FeedForward,
            ),
            vocab_size=10,
            num_layers=1,
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
        logits, _, _ = model(input_ids, attention_mask=attention_mask)
        padded_logits, _, _ = model(
            padded_input_ids,
            attention_mask=padded_attention_mask,
        )

    assert torch.allclose(
        logits,
        padded_logits[:1, -input_ids.size(1) :],
        atol=1e-6,
    )
