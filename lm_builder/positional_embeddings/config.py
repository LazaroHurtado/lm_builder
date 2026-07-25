from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Type

from torch import nn

from ..utils import module_has_attr


@dataclass
class PositionalEmbeddingConfig:
    positional_embedding_type: Type
    kwargs: Dict[str, object] = field(default_factory=dict)

    def build(self, embedding_dimension: int, context_length: int):
        positional_embedding = self.positional_embedding_type(
            embedding_dimension,
            context_length,
            **self.kwargs,
        )
        return positional_embedding

    @staticmethod
    def build_config(config: dict) -> Optional[PositionalEmbeddingConfig]:
        if not isinstance(config, dict):
            raise TypeError("qk_positional_embedding must be a mapping.")

        # Imported here to avoid importing the package while it is still
        # initializing this config module.
        from lm_builder import (  # pylint: disable=import-outside-toplevel
            positional_embeddings,
        )

        config = module_has_attr(
            config,
            "type",
            primary_module=positional_embeddings,
            fallback_module=nn,
        )
        if "type" not in config:
            raise ValueError("qk_positional_embedding.type is required.")

        positional_embedding_type = config.pop("type")
        return PositionalEmbeddingConfig(
            positional_embedding_type=positional_embedding_type,
            kwargs=config,
        )
