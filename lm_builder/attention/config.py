from __future__ import annotations

import yaml

from .. import positional_embeddings
from ..utils import module_has_attr

from dataclasses import dataclass
from torch import nn
from typing import Optional


@dataclass
class AttentionConfig:
    context_length: int
    embedding_dimension: int
    num_heads: int
    kv_heads: int = 1
    with_kv_cache: bool = False
    bias: bool = False
    attn_dropout: float = 0.0
    resid_dropout: float = 0.0
    positional_embedding: Optional[nn.Module] = None
    pos_emb_freq: float = 10_000.0

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
