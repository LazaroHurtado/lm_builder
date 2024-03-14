from __future__ import annotations

import yaml

from ..utils import module_has_attr

from dataclasses import dataclass
from torch import nn


@dataclass
class FeedForwardConfig:
    embedding_dimension: int
    intermediate_dimension: int
    activation_fn: nn.Module = nn.GELU()
    dropout: float = 0.0
    bias: bool = False
    num_experts: int = 0
    top_k: int = 0

    @staticmethod
    def from_yml(file: str) -> FeedForwardConfig:
        with open(file, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

            if "feed_forward_config" in config:
                config = config["feed_forward_config"]
            elif "ffn_config" in config:
                config = config["ffn_config"]

            return FeedForwardConfig.build_config(config)

    @staticmethod
    def build_config(config: dict) -> FeedForwardConfig:
        config = module_has_attr(config, "activation_fn", nn)

        return FeedForwardConfig(**config)
