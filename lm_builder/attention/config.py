from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from typing import List, Optional, Type

from torch import nn

from .. import positional_embeddings
from ..normalizers import NormalizerConfig
from ..utils import is_positive_integer, load_yml, module_has_attr


def _filter_config_fields(config, config_type):
    valid_fields = {config_field.name for config_field in fields(config_type)}
    return {key: value for key, value in config.items() if key in valid_fields}


def _resolve_ratio(layers, ratio):
    if ratio is None:
        if len(layers) > 1:
            raise ValueError("attention_config.ratio is required for multiple layers.")
        return [1]

    if (
        not isinstance(ratio, list)
        or len(ratio) != len(layers)
        or any(not is_positive_integer(count) for count in ratio)
    ):
        raise ValueError(
            "attention_config.ratio must contain one positive integer "
            "for each layer."
        )
    return ratio


def _merge_layer_config(shared_config, layer):
    if not isinstance(layer, dict):
        raise TypeError("Each attention layer must be a mapping.")
    if "type" not in layer:
        raise ValueError("attention_config.layers.type is required.")

    layer = dict(layer)
    merged_config = dict(shared_config)
    for norm_name in ("norm", "qk_norm"):
        if norm_name not in layer:
            continue

        layer_norm = layer.pop(norm_name)
        if norm_name == "qk_norm" and layer_norm is None:
            merged_config[norm_name] = None
            continue

        shared_norm = shared_config.get(norm_name, {}) or {}
        if not isinstance(shared_norm, dict) or not isinstance(layer_norm, dict):
            raise TypeError(f"Attention {norm_name} overrides must be mappings.")

        merged_norm = dict(shared_norm)
        merged_norm.update(layer_norm)
        merged_config[norm_name] = merged_norm

    merged_config.update(layer)
    return merged_config


@dataclass
class AttentionConfig:
    context_length: int
    embedding_dimension: int
    num_heads: int
    attention_type: Optional[Type[nn.Module]] = None
    kv_heads: int = 1
    window_size: Optional[int] = None
    bias: bool = False
    attn_dropout: float = 0.0
    resid_dropout: float = 0.0
    positional_embedding: Optional[Type[nn.Module]] = None
    inv_freq: float = 10_000.0
    norm: NormalizerConfig = field(default_factory=NormalizerConfig)
    qk_norm: Optional[NormalizerConfig] = None

    def __post_init__(self):
        if self.attention_type is None:
            raise ValueError("attention_config.layers.type is required.")
        if self.window_size is not None and not is_positive_integer(self.window_size):
            raise ValueError("window_size must be a positive integer or None.")

    def clone(self) -> AttentionConfig:
        return replace(
            self,
            norm=self.norm.clone(),
            qk_norm=None if self.qk_norm is None else self.qk_norm.clone(),
        )

    @staticmethod
    def from_yml(file: str) -> List[AttentionConfig]:
        config = load_yml(file)
        if not isinstance(config, dict):
            raise TypeError("Model config must be a mapping.")

        return AttentionConfig.build_configs(
            config["attention_config"],
            config["num_layers"],
            config["context_length"],
            config["embedding_dimension"],
        )

    @staticmethod
    def build_config(config: dict) -> AttentionConfig:
        if not isinstance(config, dict):
            raise TypeError("Attention config must be a mapping.")

        config = dict(config)

        # Imported here to avoid importing the package while it is still
        # initializing this config module.
        from lm_builder import attention  # pylint: disable=import-outside-toplevel

        config = module_has_attr(
            config,
            "type",
            primary_module=attention,
            fallback_module=nn,
        )
        if "type" in config:
            config["attention_type"] = config["type"]

        config = module_has_attr(
            config, "positional_embedding", positional_embeddings, nn
        )
        config["norm"] = NormalizerConfig.build_config(config.get("norm"))
        if config.get("qk_norm") is not None:
            config["qk_norm"] = NormalizerConfig.build_config(config["qk_norm"])

        config = _filter_config_fields(config, AttentionConfig)

        return AttentionConfig(**config)

    @staticmethod
    def build_configs(
        config: dict,
        num_layers: int,
        context_length: int,
        embedding_dimension: int,
    ) -> List[AttentionConfig]:
        if not isinstance(config, dict):
            raise TypeError("attention_config must be a mapping.")
        if not is_positive_integer(num_layers):
            raise ValueError("num_layers must be a positive integer.")

        config = dict(config)
        layers = config.get("layers")
        if not isinstance(layers, list) or not layers:
            raise ValueError("attention_config.layers must be a non-empty list.")
        ratio = _resolve_ratio(layers, config.get("ratio"))

        shared_config = {
            key: value
            for key, value in config.items()
            if key not in {"layers", "ratio"}
        }
        resolved_layers = []
        for layer in layers:
            layer_config = _merge_layer_config(shared_config, layer)
            layer_config["context_length"] = context_length
            layer_config["embedding_dimension"] = embedding_dimension
            resolved_layers.append(AttentionConfig.build_config(layer_config))

        pattern = [
            layer
            for layer, layer_count in zip(resolved_layers, ratio)
            for _ in range(layer_count)
        ]
        if num_layers % len(pattern) != 0:
            raise ValueError(
                "num_layers must be divisible by the sum of attention_config.ratio."
            )

        return [
            pattern[layer_index % len(pattern)].clone()
            for layer_index in range(num_layers)
        ]
