from __future__ import annotations

import yaml

from .. import attention
from .. import ffn
from .. import normalizers
from .. import positional_embeddings
from ..utils import module_has_attr

from dataclasses import dataclass
from torch import nn
from typing import Optional


@dataclass
class TransformerConfig:
    attention_config: attention.AttentionConfig
    ffn_config: ffn.FeedForwardConfig
    vocab_size: int
    num_layers: int
    attention: Optional[attention.Attention] = None
    ffn: Optional[ffn.FeedForward] = None
    norm: nn.Module = nn.LayerNorm
    attn_norm: nn.Module = nn.LayerNorm
    ffn_norm: nn.Module = nn.LayerNorm
    token_embedding: nn.Module = nn.Embedding
    positional_embedding: Optional[nn.Module] = None
    inv_freq: float = 10_000.0
    bias: bool = False
    norm_bias: bool = False
    dropout: float = 0.0

    @staticmethod
    def from_yml(file: str) -> TransformerConfig:
        with open(file, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

            return TransformerConfig.build_config(config)

    @staticmethod
    def build_config(config: dict) -> TransformerConfig:
        config = module_has_attr(
            config, "attention", primary_module=attention, fallback_module=nn
        )

        config = module_has_attr(config, "ffn", primary_module=ffn, fallback_module=nn)

        # pylint: disable=duplicate-code
        config = module_has_attr(
            config,
            "positional_embedding",
            primary_module=positional_embeddings,
            fallback_module=nn,
        )

        config = module_has_attr(config, "token_embedding", nn)
        config = module_has_attr(
            config, "norm", primary_module=normalizers, fallback_module=nn
        )
        config = module_has_attr(
            config,
            "attn_norm",
            primary_module=normalizers,
            fallback_module=nn,
        )
        config = module_has_attr(
            config, "ffn_norm", primary_module=normalizers, fallback_module=nn
        )

        config["attention_config"] = attention.AttentionConfig.build_config(
            config["attention_config"]
        )
        config["ffn_config"] = ffn.FeedForwardConfig.build_config(config["ffn_config"])

        return TransformerConfig(**config)
