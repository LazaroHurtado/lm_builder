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


def test_absolute_pe_instances_are_independent():
    first = AbsolutePE(CONTEXT_LENGTH, EMBEDDING_DIM, BASE)
    second = AbsolutePE(CONTEXT_LENGTH, EMBEDDING_DIM, BASE)

    assert first is not second
    assert first.weight.data_ptr() != second.weight.data_ptr()


def test_absolute_pe_uses_a_constant_non_trainable_table():
    positional_embedding = AbsolutePE(CONTEXT_LENGTH, EMBEDDING_DIM, BASE)
    x = torch.zeros(2, 4, EMBEDDING_DIM, dtype=torch.float16)

    output = positional_embedding(x)

    assert positional_embedding.weight.device.type == "cpu"
    assert positional_embedding.weight.dtype == torch.float32
    assert not positional_embedding.weight.requires_grad
    torch.testing.assert_close(
        output[0],
        positional_embedding.weight[:4].to(dtype=x.dtype),
    )


def test_absolute_pe_supports_explicit_positions():
    positional_embedding = AbsolutePE(CONTEXT_LENGTH, EMBEDDING_DIM, BASE)
    x = torch.zeros(2, 3, EMBEDDING_DIM)
    position_ids = torch.tensor([[0, 3, 5], [1, 2, 4]])

    output = positional_embedding(x, position_ids=position_ids)

    torch.testing.assert_close(output, positional_embedding.weight[position_ids])


def test_absolute_pe_is_a_nonpersistent_model_buffer():
    model = build_transformer(AbsolutePE)

    assert isinstance(model.wpe, torch.nn.Module)
    assert "weight" in dict(model.wpe.named_buffers())
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
