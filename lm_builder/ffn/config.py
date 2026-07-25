from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Type

from torch import nn

from ..normalizers import NormalizerConfig
from ..utils import module_has_attr


@dataclass
class FeedForwardConfig:
    embedding_dimension: int
    intermediate_dimension: int
    ffn_type: Type[nn.Module]
    activation_fn: nn.Module = nn.GELU
    dropout: float = 0.0
    bias: bool = False
    num_experts: int = 0
    top_k: int = 0
    num_shared_experts: int = 0
    norm: NormalizerConfig = field(default_factory=NormalizerConfig)

    def clone(self) -> FeedForwardConfig:
        return replace(
            self,
            norm=self.norm.clone(),
        )

    @staticmethod
    def build_config(
        config: dict,
        embedding_dimension: int,
    ) -> FeedForwardConfig:
        config = dict(config)

        # Imported here to avoid importing the package while it is still
        # initializing this config module.
        from lm_builder import ffn  # pylint: disable=import-outside-toplevel

        config = module_has_attr(
            config,
            "type",
            primary_module=ffn,
            fallback_module=nn,
        )
        config = module_has_attr(config, "activation_fn", nn)
        if "type" not in config:
            raise ValueError("ffn_config.type is required.")
        config["ffn_type"] = config.pop("type")
        config["embedding_dimension"] = embedding_dimension
        config["norm"] = NormalizerConfig.build_config(config.get("norm"))

        return FeedForwardConfig(**config)
