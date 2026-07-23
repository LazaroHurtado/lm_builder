import torch

from lm_builder.attention import (
    AttentionConfig,
    AttentionLayerConfig,
    CausalMultiHeadAttention,
)
from lm_builder.ffn import FeedForward, FeedForwardConfig
from lm_builder.positional_embeddings import AbsolutePE, LearnablePE
from lm_builder.transformer import Transformer, TransformerConfig

CONTEXT_LENGTH = 8
EMBEDDING_DIM = 6
BASE = 10_000.0


def setup_function():
    AbsolutePE.KEY_TO_INSTANCE.clear()


def build_transformer(positional_embedding):
    return Transformer(
        TransformerConfig(
            embedding_dimension=EMBEDDING_DIM,
            context_length=CONTEXT_LENGTH,
            attention_config=AttentionConfig(
                qk_positional_embedding=None,
                layers=[
                    AttentionLayerConfig(
                        context_length=CONTEXT_LENGTH,
                        embedding_dimension=EMBEDDING_DIM,
                        num_heads=2,
                        attention_type=CausalMultiHeadAttention,
                    )
                ],
            ),
            ffn_config=FeedForwardConfig(
                embedding_dimension=EMBEDDING_DIM,
                intermediate_dimension=12,
                ffn_type=FeedForward,
            ),
            vocab_size=10,
            num_layers=1,
            positional_embedding=positional_embedding,
        )
    )


class RecordingLearnablePE(LearnablePE):
    def forward(self, input, position_ids=None):  # pylint: disable=redefined-builtin
        self.last_position_ids = position_ids
        return super().forward(input, position_ids=position_ids)


def test_absolute_pe_is_reused_by_configuration():
    positional_embedding = AbsolutePE(CONTEXT_LENGTH, EMBEDDING_DIM, BASE)

    assert AbsolutePE(CONTEXT_LENGTH, EMBEDDING_DIM, BASE) is positional_embedding
    assert (
        AbsolutePE(CONTEXT_LENGTH + 1, EMBEDDING_DIM, BASE) is not positional_embedding
    )
    assert (
        AbsolutePE(CONTEXT_LENGTH, EMBEDDING_DIM + 2, BASE) is not positional_embedding
    )
    assert (
        AbsolutePE(CONTEXT_LENGTH, EMBEDDING_DIM, BASE + 1) is not positional_embedding
    )


def test_absolute_pe_generates_a_constant_table_lazily():
    positional_embedding = AbsolutePE(CONTEXT_LENGTH, EMBEDDING_DIM, BASE)
    x = torch.zeros(2, 4, EMBEDDING_DIM, dtype=torch.float16)

    assert positional_embedding.weight is None

    output = positional_embedding(x)

    assert positional_embedding.weight.device == x.device
    assert positional_embedding.weight.dtype == x.dtype
    assert not positional_embedding.weight.requires_grad
    torch.testing.assert_close(output[0], positional_embedding.weight[:4])

    shared_embedding = AbsolutePE(CONTEXT_LENGTH, EMBEDDING_DIM, BASE)
    assert shared_embedding.weight is positional_embedding.weight


def test_absolute_pe_supports_explicit_positions():
    positional_embedding = AbsolutePE(CONTEXT_LENGTH, EMBEDDING_DIM, BASE)
    x = torch.zeros(2, 3, EMBEDDING_DIM)
    position_ids = torch.tensor([[0, 3, 5], [1, 2, 4]])

    output = positional_embedding(x, position_ids=position_ids)

    torch.testing.assert_close(output, positional_embedding.weight[position_ids])


def test_absolute_pe_is_not_registered_as_model_state():
    model = build_transformer(AbsolutePE)

    assert not isinstance(model.wpe, torch.nn.Module)
    assert "wpe.weight" not in model.state_dict()


def test_learnable_pe_is_an_embedding_parameter():
    model = build_transformer(LearnablePE)

    assert isinstance(model.wpe, torch.nn.Embedding)
    assert model.wpe.weight.requires_grad
    assert "wpe.weight" in model.state_dict()


def test_learnable_pe_supports_explicit_positions():
    positional_embedding = LearnablePE(CONTEXT_LENGTH, EMBEDDING_DIM, BASE)
    x = torch.zeros(2, 3, EMBEDDING_DIM)
    position_ids = torch.tensor([[0, 3, 5], [1, 2, 4]])

    output = positional_embedding(x, position_ids=position_ids)

    torch.testing.assert_close(output, positional_embedding.weight[position_ids])


def test_transformer_generates_implicit_input_positions():
    model = build_transformer(RecordingLearnablePE)

    model(torch.tensor([[1, 2, 3], [4, 5, 6]]))

    torch.testing.assert_close(
        model.wpe.last_position_ids,
        torch.tensor([[0, 1, 2], [0, 1, 2]]),
    )


def test_transformer_passes_explicit_input_positions():
    model = build_transformer(RecordingLearnablePE)
    position_ids = torch.tensor([[0, 2, 3], [1, 2, 4]])

    model(
        torch.tensor([[1, 2, 3], [4, 5, 6]]),
        position_ids=position_ids,
    )

    torch.testing.assert_close(model.wpe.last_position_ids, position_ids)


def test_gpt2_uses_learnable_pe():
    config = TransformerConfig.from_yml("examples/gpt2_xl.yml")

    assert config.positional_embedding is LearnablePE
