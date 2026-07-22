from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional, Type

from torch import nn

from ..normalizers import NormalizerConfig
from ..utils import load_yml, module_has_attr


@dataclass
class FeedForwardConfig:
    embedding_dimension: int
    intermediate_dimension: int
    ffn_type: Optional[Type[nn.Module]] = None
    activation_fn: nn.Module = nn.GELU
    dropout: float = 0.0
    bias: bool = False
    num_experts: int = 0
    top_k: int = 0
    norm: NormalizerConfig = field(default_factory=NormalizerConfig)

    def __post_init__(self):
        if self.ffn_type is None:
            raise ValueError("ffn_config.type is required.")

    def clone(self) -> FeedForwardConfig:
        return replace(
            self,
            norm=self.norm.clone(),
        )

    @staticmethod
    def from_yml(file: str) -> FeedForwardConfig:
        config = load_yml(file)

        return FeedForwardConfig.build_config(
            config["ffn_config"],
            config["embedding_dimension"],
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
        ffn_type = config.pop("type", None)
        config["ffn_type"] = ffn_type
        config["embedding_dimension"] = embedding_dimension
        config["norm"] = NormalizerConfig.build_config(config.get("norm"))

        return FeedForwardConfig(**config)
