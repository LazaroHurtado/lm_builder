from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Type

from torch import nn

from lm_builder import attention, ffn, normalizers, positional_embeddings
from lm_builder.utils import is_positive_integer, load_yml, module_has_attr


@dataclass
class TransformerConfig:
    embedding_dimension: int
    context_length: int
    attention_config: List[attention.AttentionConfig]
    ffn_config: ffn.FeedForwardConfig
    vocab_size: int
    num_layers: int
    norm: normalizers.NormalizerConfig = field(
        default_factory=normalizers.NormalizerConfig
    )
    token_embedding: Type[nn.Module] = nn.Embedding
    positional_embedding: Optional[Type] = None
    inv_freq: float = 10_000.0
    bias: bool = False
    dropout: float = 0.0
    tie_word_embeddings: bool = False

    def __post_init__(self):
        if not is_positive_integer(self.embedding_dimension):
            raise ValueError("embedding_dimension must be a positive integer.")
        if not is_positive_integer(self.context_length):
            raise ValueError("context_length must be a positive integer.")
        if not is_positive_integer(self.num_layers):
            raise ValueError("num_layers must be a positive integer.")
        if len(self.attention_config) != self.num_layers:
            raise ValueError(
                "attention_config must contain one AttentionConfig per layer."
            )

    @staticmethod
    def from_yml(file: str) -> TransformerConfig:
        return TransformerConfig.build_config(load_yml(file))

    @staticmethod
    def build_config(config: dict) -> TransformerConfig:
        config = dict(config)

        config = module_has_attr(
            config, "positional_embedding", positional_embeddings, nn
        )
        config = module_has_attr(config, "token_embedding", nn)

        config["attention_config"] = attention.AttentionConfig.build_configs(
            config["attention_config"],
            config["num_layers"],
            config["context_length"],
            config["embedding_dimension"],
        )
        config["ffn_config"] = ffn.FeedForwardConfig.build_config(
            config["ffn_config"],
            config["embedding_dimension"],
        )
        config["norm"] = normalizers.NormalizerConfig.build_config(config.get("norm"))

        return TransformerConfig(**config)
