import pytest
from torch import nn

from lm_builder.normalizers import NormalizerConfig, RMSNorm


def test_normalizer_config_resolves_type_and_inline_options():
    config = NormalizerConfig.build_config(
        {
            "type": "RMSNorm",
            "eps": 1e-5,
            "custom_option": "preserved",
        }
    )

    assert config.normalizer_type is RMSNorm
    assert config.kwargs == {
        "eps": 1e-5,
        "custom_option": "preserved",
        "bias": False,
    }


def test_normalizer_config_builds_torch_normalizer():
    config = NormalizerConfig.build_config(
        {
            "type": "LayerNorm",
            "eps": 1e-4,
            "elementwise_affine": False,
            "bias": False,
        }
    )

    normalizer = config.build(8)

    assert isinstance(normalizer, nn.LayerNorm)
    assert normalizer.eps == 1e-4
    assert not normalizer.elementwise_affine


def test_normalizer_config_defaults_to_biasless_layer_norm():
    config = NormalizerConfig.build_config()

    normalizer = config.build(8)

    assert isinstance(normalizer, nn.LayerNorm)
    assert normalizer.bias is None


def test_normalizer_config_requires_mapping():
    with pytest.raises(TypeError, match="norm must be a mapping"):
        NormalizerConfig.build_config("RMSNorm")
