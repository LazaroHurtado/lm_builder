import functools

import pytest
from torch import nn

from lm_builder.positional_embeddings import (
    PositionalEmbeddingConfig,
    RotaryPE,
)


def test_build_config_resolves_type_and_constructor_kwargs():
    config = PositionalEmbeddingConfig.build_config(
        {
            "type": "RotaryPE",
            "base": 1_000_000.0,
        }
    )

    assert config.positional_embedding_type is RotaryPE
    assert config.kwargs == {"base": 1_000_000.0}

    positional_embedding = config.build(
        embedding_dimension=4,
        context_length=8,
    )
    assert isinstance(positional_embedding, RotaryPE)
    assert positional_embedding.embedding_dim == 4
    assert positional_embedding.context_len == 8
    assert positional_embedding.base == 1_000_000.0


def test_build_returns_independent_positional_embeddings():
    config = PositionalEmbeddingConfig.build_config({"type": "RotaryPE"})

    first = config.build(embedding_dimension=4, context_length=8)
    second = config.build(embedding_dimension=4, context_length=8)

    assert first is not second


def test_build_config_requires_mapping():
    with pytest.raises(TypeError, match="must be a mapping"):
        PositionalEmbeddingConfig.build_config("RotaryPE")


def test_build_config_requires_type():
    with pytest.raises(ValueError, match="type is required"):
        PositionalEmbeddingConfig.build_config({"base": 1_000_000.0})


def test_type_resolution_retains_torch_nn_fallback(monkeypatch):
    class TwoPhaseIdentity(nn.Identity):
        def prepare(self, *_args):
            return None

        @staticmethod
        def apply_qk(query, key, _position_data):
            return query, key

    monkeypatch.setattr(nn, "TwoPhaseIdentity", TwoPhaseIdentity, raising=False)
    config = PositionalEmbeddingConfig.build_config({"type": "TwoPhaseIdentity"})

    assert isinstance(config.build(4, 8), TwoPhaseIdentity)


def test_build_does_not_require_attention_specific_interface():
    config = PositionalEmbeddingConfig.build_config({"type": "Identity"})

    assert isinstance(config.build(4, 8), nn.Identity)


def test_build_supports_stateful_applicator():
    class StatefulApplicator(nn.Module):
        def __init__(self, *_args):
            super().__init__()

        def prepare(self, *_args):
            return None

        def apply_qk(self, query, key, _position_data):
            return query, key

    config = PositionalEmbeddingConfig(positional_embedding_type=StatefulApplicator)

    assert isinstance(config.build(4, 8), StatefulApplicator)


def test_build_supports_wrapped_instance_applicator():
    class WrappedApplicator(nn.Module):
        def __init__(self, *_args):
            super().__init__()
            self.apply_qk = functools.partial(self._apply_qk)

        def prepare(self, *_args):
            return None

        def _apply_qk(self, query, key, _position_data):
            return query, key

    config = PositionalEmbeddingConfig(positional_embedding_type=WrappedApplicator)

    assert isinstance(config.build(4, 8), WrappedApplicator)
