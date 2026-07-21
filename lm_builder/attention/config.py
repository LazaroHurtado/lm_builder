from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import yaml
from torch import nn

from .. import positional_embeddings
from ..utils import module_has_attr


@dataclass
class AttentionConfig:
    context_length: int
    embedding_dimension: int
    num_heads: int
    kv_heads: int = 1
    bias: bool = False
    attn_dropout: float = 0.0
    resid_dropout: float = 0.0
    positional_embedding: Optional[nn.Module] = None
    inv_freq: float = 10_000.0
    window_size: Optional[int] = None
    attention_ratio: Optional[str] = None

    def __post_init__(self):
        self.get_attention_ratio()

    def get_attention_ratio(self):
        if self.attention_ratio is None:
            return None

        if not isinstance(self.attention_ratio, str):
            raise ValueError(
                "attention_ratio must contain at least two colon-separated "
                "positive integers."
            )

        counts = self.attention_ratio.split(":")
        if (
            len(counts) < 2
            or any(not count.isdecimal() for count in counts)
            or any(int(count) <= 0 for count in counts)
        ):
            raise ValueError(
                "attention_ratio must contain at least two colon-separated "
                "positive integers."
            )

        return tuple(int(count) for count in counts)

    @staticmethod
    def from_yml(file: str) -> AttentionConfig:
        with open(file, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            if "attention_config" in config:
                config = config["attention_config"]

            return AttentionConfig.build_config(config)

    @staticmethod
    def build_config(config: dict) -> AttentionConfig:
        # pylint: disable=duplicate-code
        config = module_has_attr(
            config,
            "positional_embedding",
            primary_module=positional_embeddings,
            fallback_module=nn,
        )

        return AttentionConfig(**config)
