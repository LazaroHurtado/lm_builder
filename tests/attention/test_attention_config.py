import pytest
from torch import nn

from lm_builder.attention import (
    AttentionConfig,
    CausalMultiHeadAttention,
    GroupedQueryAttention,
    MultiHeadAttention,
)
from lm_builder.normalizers import RMSNorm


def build_attention_configs(num_layers=1, **overrides):
    config = {
        "num_heads": 4,
        "norm": {
            "type": "LayerNorm",
            "eps": 1e-5,
            "bias": False,
        },
        "layers": [
            {
                "type": "GroupedQueryAttention",
                "kv_heads": 2,
                "window_size": 4,
            }
        ],
    }
    config.update(overrides)
    return AttentionConfig.build_configs(
        config,
        num_layers,
        context_length=8,
        embedding_dimension=16,
    )


def test_build_configs_resolves_shared_values_and_layer_overrides():
    configs = build_attention_configs(
        num_layers=3,
        ratio=[2, 1],
        layers=[
            {
                "type": "GroupedQueryAttention",
                "num_heads": 2,
                "kv_heads": 2,
                "window_size": 4,
                "norm": {"eps": 1e-4},
            },
            {
                "type": "CausalMultiHeadAttention",
                "positional_embedding": None,
            },
        ],
    )

    grouped_query, _, causal = configs
    assert grouped_query.attention_type is GroupedQueryAttention
    assert grouped_query.context_length == 8
    assert grouped_query.embedding_dimension == 16
    assert grouped_query.num_heads == 2
    assert grouped_query.kv_heads == 2
    assert grouped_query.window_size == 4
    assert grouped_query.norm.normalizer_type is nn.LayerNorm
    assert grouped_query.norm.kwargs == {
        "eps": 1e-4,
        "bias": False,
    }

    assert causal.attention_type is CausalMultiHeadAttention
    assert causal.num_heads == 4
    assert causal.window_size is None
    assert causal.norm.kwargs == {
        "eps": 1e-5,
        "bias": False,
    }


def test_build_configs_expands_ratio_into_isolated_layer_configs():
    configs = build_attention_configs(
        num_layers=6,
        ratio=[2, 1],
        layers=[
            {
                "type": "GroupedQueryAttention",
                "kv_heads": 2,
                "window_size": 4,
            },
            {"type": "CausalMultiHeadAttention"},
        ],
    )

    assert [config.attention_type for config in configs] == [
        GroupedQueryAttention,
        GroupedQueryAttention,
        CausalMultiHeadAttention,
        GroupedQueryAttention,
        GroupedQueryAttention,
        CausalMultiHeadAttention,
    ]
    assert [config.window_size for config in configs] == [
        4,
        4,
        None,
        4,
        4,
        None,
    ]
    assert len({id(config) for config in configs}) == 6
    assert len({id(config.norm) for config in configs}) == 6
    assert len({id(config.norm.kwargs) for config in configs}) == 6


def test_build_configs_merges_and_clones_qk_norm_config():
    configs = build_attention_configs(
        num_layers=2,
        qk_norm={
            "type": "RMSNorm",
            "eps": 1e-5,
        },
        layers=[
            {
                "type": "GroupedQueryAttention",
                "kv_heads": 2,
                "qk_norm": {"eps": 1e-6},
            }
        ],
    )

    assert all(config.qk_norm.normalizer_type is RMSNorm for config in configs)
    assert all(
        config.qk_norm.kwargs == {"eps": 1e-6, "bias": False} for config in configs
    )
    assert len({id(config.qk_norm) for config in configs}) == 2
    assert len({id(config.qk_norm.kwargs) for config in configs}) == 2


def test_attention_layer_can_disable_shared_qk_norm():
    configs = build_attention_configs(
        num_layers=2,
        qk_norm={"type": "RMSNorm"},
        ratio=[1, 1],
        layers=[
            {
                "type": "GroupedQueryAttention",
                "kv_heads": 2,
            },
            {
                "type": "CausalMultiHeadAttention",
                "qk_norm": None,
            },
        ],
    )

    assert configs[0].qk_norm.normalizer_type is RMSNorm
    assert configs[1].qk_norm is None


def test_qk_norm_is_disabled_by_default():
    configs = build_attention_configs(num_layers=2)

    assert all(config.qk_norm is None for config in configs)


def test_single_attention_layer_does_not_require_ratio():
    configs = build_attention_configs(num_layers=3)

    assert len(configs) == 3
    assert all(config.attention_type is GroupedQueryAttention for config in configs)


def test_attention_layer_requires_its_own_type():
    with pytest.raises(
        ValueError,
        match="attention_config.layers.type is required",
    ):
        build_attention_configs(
            type="CausalMultiHeadAttention",
            layers=[
                {
                    "kv_heads": 2,
                }
            ],
        )


def test_single_attention_layer_allows_ratio():
    configs = build_attention_configs(num_layers=4, ratio=[2])

    assert len(configs) == 4


def test_multiple_attention_layers_require_ratio():
    with pytest.raises(
        ValueError,
        match="ratio is required for multiple layers",
    ):
        build_attention_configs(
            layers=[
                {"type": "GroupedQueryAttention", "kv_heads": 2},
                {"type": "CausalMultiHeadAttention"},
            ]
        )


@pytest.mark.parametrize(
    "ratio",
    [
        "2:1",
        [1],
        [1, 0],
        [1, -1],
        [1, 1.5],
        [1, True],
    ],
)
def test_ratio_requires_one_positive_integer_per_layer(ratio):
    with pytest.raises(
        ValueError,
        match="ratio must contain one positive integer for each layer",
    ):
        build_attention_configs(
            ratio=ratio,
            layers=[
                {"type": "GroupedQueryAttention", "kv_heads": 2},
                {"type": "CausalMultiHeadAttention"},
            ],
        )


def test_ratio_sum_must_divide_num_layers():
    with pytest.raises(
        ValueError,
        match="num_layers must be divisible by the sum",
    ):
        build_attention_configs(
            num_layers=4,
            ratio=[2, 1],
            layers=[
                {"type": "GroupedQueryAttention", "kv_heads": 2},
                {"type": "CausalMultiHeadAttention"},
            ],
        )


@pytest.mark.parametrize(
    ("field", "expected"),
    [
        ("context_length", 8),
        ("embedding_dimension", 16),
    ],
)
def test_nested_dimensions_are_ignored(field, expected):
    configs = build_attention_configs(
        **{
            field: 2,
            "layers": [
                {
                    "type": "GroupedQueryAttention",
                    "kv_heads": 2,
                    field: 4,
                }
            ],
        }
    )

    assert getattr(configs[0], field) == expected


def test_outer_attention_options_are_shared_defaults():
    configs = build_attention_configs(
        kv_heads=3,
        window_size=2,
        layers=[{"type": "CausalMultiHeadAttention"}],
    )

    assert configs[0].kv_heads == 3
    assert configs[0].window_size == 2


def test_unknown_attention_options_are_ignored():
    configs = build_attention_configs(
        unknown_shared_option=True,
        layers=[
            {
                "type": "GroupedQueryAttention",
                "kv_heads": 2,
                "unknown_layer_option": True,
            }
        ],
    )

    assert not hasattr(configs[0], "unknown_shared_option")
    assert not hasattr(configs[0], "unknown_layer_option")


def test_noncausal_attention_rejects_window_size():
    config = build_attention_configs(
        layers=[
            {
                "type": "MultiHeadAttention",
                "window_size": 4,
            }
        ]
    )[0]

    with pytest.raises(
        ValueError,
        match="MultiHeadAttention does not support window_size",
    ):
        config.attention_type(config)


def test_kv_heads_is_allowed_on_attention_types_that_do_not_use_it():
    configs = build_attention_configs(
        layers=[
            {
                "type": "CausalMultiHeadAttention",
                "kv_heads": 2,
            }
        ]
    )

    assert configs[0].kv_heads == 2


def test_noncausal_attention_cannot_opt_into_window_size():
    class NonCausalWindowAttention(MultiHeadAttention):
        supports_window_size = True

    config = AttentionConfig(
        context_length=8,
        embedding_dimension=16,
        num_heads=4,
        attention_type=NonCausalWindowAttention,
        window_size=4,
    )

    with pytest.raises(
        ValueError,
        match="NonCausalWindowAttention does not support window_size",
    ):
        config.attention_type(config)


def test_attention_type_retains_torch_nn_fallback():
    configs = build_attention_configs(
        layers=[
            {
                "type": "Identity",
            }
        ]
    )

    assert configs[0].attention_type is nn.Identity


def test_from_yml_returns_one_resolved_config_per_transformer_layer(tmp_path):
    config_path = tmp_path / "model.yml"
    config_path.write_text(
        """
context_length: 8
embedding_dimension: 16
attention_config:
  num_heads: 4
  ratio: [2, 1]
  layers:
    - type: GroupedQueryAttention
      kv_heads: 2
    - type: CausalMultiHeadAttention
num_layers: 3
""",
        encoding="utf-8",
    )

    configs = AttentionConfig.from_yml(config_path)

    assert [config.attention_type for config in configs] == [
        GroupedQueryAttention,
        GroupedQueryAttention,
        CausalMultiHeadAttention,
    ]
