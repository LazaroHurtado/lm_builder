import pytest
import torch
from torch.nn import functional as F

from lm_builder.attention import (
    AttentionConfig,
    AttentionLayerConfig,
    CausalMultiHeadAttention,
)
from lm_builder.ffn import FeedForward, FeedForwardConfig
from lm_builder.normalizers import RMSNorm
from lm_builder.positional_embeddings import (
    AbsolutePE,
    PositionalEmbeddingConfig,
    RotaryPE,
)
from lm_builder.transformer import Transformer, TransformerConfig


class RecordingQKPositionalEmbedding(torch.nn.Module):
    def __init__(self, *_args):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(()))
        self.prepare_calls = 0
        self.apply_qk_calls = 0
        self.position_ids = None

    def prepare(self, _x, position_ids):
        self.prepare_calls += 1
        self.position_ids = position_ids.clone()
        return self

    @staticmethod
    def apply_qk(query, key, position_data):
        position_data.apply_qk_calls += 1
        return query * position_data.scale, key * position_data.scale


def build_attention_configs(
    num_layers=1,
    qk_positional_embedding=None,
    attention_type=CausalMultiHeadAttention,
    **kwargs,
):
    return AttentionConfig(
        qk_positional_embedding=qk_positional_embedding,
        layers=[
            AttentionLayerConfig(
                attention_type=attention_type,
                **kwargs,
            )
            for _ in range(num_layers)
        ],
    )


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


def test_qk_positional_embedding_prepares_once_and_applies_per_layer():
    positional_embedding_config = PositionalEmbeddingConfig(
        positional_embedding_type=RecordingQKPositionalEmbedding
    )
    config = TransformerConfig(
        embedding_dimension=8,
        context_length=4,
        attention_config=build_attention_configs(
            num_layers=3,
            qk_positional_embedding=positional_embedding_config,
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
        num_layers=3,
    )
    model = Transformer(config)
    second_model = Transformer(config)

    model_position = model.qk_positional_embedding
    model(torch.tensor([[1, 2, 3], [4, 5, 6]]))

    assert model_position is not second_model.qk_positional_embedding
    assert (
        model_position.scale.data_ptr()
        != second_model.qk_positional_embedding.scale.data_ptr()
    )
    assert model_position.prepare_calls == 1
    assert model_position.apply_qk_calls == 3
    assert torch.equal(
        model_position.position_ids,
        torch.tensor([[0, 1, 2], [0, 1, 2]]),
    )
    visited_modules = []
    model.apply(visited_modules.append)
    assert model_position in visited_modules
    assert all(
        block.attn.qk_positional_embedding is model_position for block in model.blocks
    )
    assert {key for key in model.state_dict() if key.endswith("scale")} == {
        "qk_positional_embedding.scale",
        "blocks.0.attn.qk_positional_embedding.scale",
        "blocks.1.attn.qk_positional_embedding.scale",
        "blocks.2.attn.qk_positional_embedding.scale",
    }


def test_custom_attention_constructor_accepts_qk_positional_embedding():
    class CustomAttention(CausalMultiHeadAttention):
        def __init__(self, config, qk_positional_embedding=None):
            super().__init__(
                config,
                qk_positional_embedding=qk_positional_embedding,
            )

    model = Transformer(
        TransformerConfig(
            embedding_dimension=8,
            context_length=4,
            attention_config=build_attention_configs(
                attention_type=CustomAttention,
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

    assert isinstance(model.blocks[0].attn, CustomAttention)


def test_masked_rotary_forward_compiles_as_one_full_graph():
    model = Transformer(
        TransformerConfig(
            embedding_dimension=8,
            context_length=4,
            attention_config=build_attention_configs(
                qk_positional_embedding=PositionalEmbeddingConfig(
                    positional_embedding_type=RotaryPE
                ),
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
    ).eval()
    input_ids = torch.tensor([[1, 2, 3, 4]])
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

    expected = model(input_ids, attention_mask=attention_mask)
    compiled = torch.compile(model, backend="eager", fullgraph=True)
    actual = compiled(input_ids, attention_mask=attention_mask)

    torch.testing.assert_close(actual[0], expected[0])


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
    block = model.blocks[0]

    assert isinstance(block.attn, CausalMultiHeadAttention)
    assert isinstance(block.ffn, FeedForward)
    assert config.attention_config.layers[0].context_length == config.context_length
    assert (
        config.attention_config.layers[0].embedding_dimension
        == config.embedding_dimension
    )
    assert config.ffn_config.embedding_dimension == config.embedding_dimension
    assert isinstance(block.attn_norm, torch.nn.LayerNorm)
    assert block.attn_norm.eps == 1e-4
    assert isinstance(block.ffn_norm, RMSNorm)
    assert block.ffn_norm.eps == 2e-5
    assert isinstance(model.norm, torch.nn.LayerNorm)
    assert model.norm.eps == 3e-5
    assert config.tie_word_embeddings
    assert model.lm_head.weight is model.wte.weight


def test_tied_word_embeddings_share_parameters_gradients_and_state():
    model = build_transformer(tie_word_embeddings=True)
    shared_weight = model.wte.weight
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
        state_dict["wte.weight"].data_ptr() == state_dict["lm_head.weight"].data_ptr()
    )


def test_word_embeddings_are_untied_by_default():
    model = build_transformer()

    assert not model.config.tie_word_embeddings
    assert model.lm_head.weight is not model.wte.weight
    assert model.lm_head.weight.data_ptr() != model.wte.weight.data_ptr()


def test_tied_word_embeddings_remain_tied_after_assigned_state_load():
    source = build_transformer(tie_word_embeddings=True)
    with torch.no_grad():
        source.wte.weight.fill_(2.5)

    model = build_transformer(tie_word_embeddings=True).to("meta")
    model.load_state_dict(source.state_dict(), assign=True)

    assert model.lm_head.weight is model.wte.weight
    assert torch.equal(
        model.wte.weight,
        torch.full_like(model.wte.weight, 2.5),
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
        match="one AttentionLayerConfig per layer",
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
        ("examples/qwen3_0_6b.yml", 28, True),
        ("examples/tinystories_200m.yml", 17, False),
    ],
)
def test_example_configs_use_resolved_module_configs(
    config_path,
    expected_layers,
    expected_tied_embeddings,
):
    config = TransformerConfig.from_yml(config_path)

    assert len(config.attention_config.layers) == expected_layers
    assert all(
        layer.attention_type is not None for layer in config.attention_config.layers
    )
    assert all(
        layer.context_length == config.context_length
        for layer in config.attention_config.layers
    )
    assert all(
        layer.embedding_dimension == config.embedding_dimension
        for layer in config.attention_config.layers
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
        qk_positional_embedding=(
            PositionalEmbeddingConfig(positional_embedding_type=RotaryPE)
            if position_type == "rotary"
            else None
        ),
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
    model.blocks[0].attn.has_flash_attn = use_scaled_dot_product_attention and hasattr(
        F, "scaled_dot_product_attention"
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
