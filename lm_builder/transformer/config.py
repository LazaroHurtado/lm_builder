from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Type, Union

import yaml
from torch import nn

from lm_builder import attention, ffn, normalizers, positional_embeddings
from lm_builder.utils import module_has_attr


@dataclass
class TransformerConfig:
    attention_config: attention.AttentionConfig
    ffn_config: ffn.FeedForwardConfig
    vocab_size: int
    num_layers: int
    attention: Optional[
        Union[
            Type[attention.Attention],
            List[Type[attention.Attention]],
        ]
    ] = None
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

    def __post_init__(self):
        attention_ratio = self.attention_config.get_attention_ratio()
        has_attention_list = isinstance(self.attention, list)

        if has_attention_list:
            if attention_ratio is None:
                raise ValueError(
                    "attention can only be a list when attention_ratio is set."
                )
            if len(self.attention) != len(attention_ratio):
                raise ValueError(
                    "attention list length must match attention_ratio component count."
                )
        elif attention_ratio is not None:
            raise ValueError("attention must be a list when attention_ratio is set.")

    @staticmethod
    def from_yml(file: str) -> TransformerConfig:
        with open(file, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

            return TransformerConfig.build_config(config)

    @staticmethod
    def _resolve_attention(config: dict) -> dict:
        if not isinstance(config.get("attention"), list):
            return module_has_attr(
                config,
                "attention",
                primary_module=attention,
                fallback_module=nn,
            )

        resolved_attention = []
        for attention_name in config["attention"]:
            resolved = module_has_attr(
                {"attention": attention_name},
                "attention",
                primary_module=attention,
                fallback_module=nn,
            )
            resolved_attention.append(resolved["attention"])

        config["attention"] = resolved_attention
        return config

    @staticmethod
    def build_config(config: dict) -> TransformerConfig:
        config = TransformerConfig._resolve_attention(config)

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
