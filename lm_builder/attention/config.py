from __future__ import annotations

import torch.nn as nn
import yaml

from .. import positional_embeddings
from ..utils import module_has_attr

from dataclasses import dataclass
from typing import Optional

@dataclass
class AttentionConfig():
    context_length: int
    embedding_dimension: int
    num_heads: int
    kv_heads: Optional[int] = 1
    with_kv_cache: Optional[bool] = False
    attn_dropout: Optional[float] = 0.0
    resid_dropout: Optional[float] = 0.0
    positional_embedding: Optional[nn.Module] = None

    @staticmethod
    def from_yml(file: str) -> AttentionConfig:
        with open(file, 'r') as f:
            config = yaml.safe_load(f)
            if "attention_config" in config:
                config = config["attention_config"]

            return AttentionConfig.build_config(config)
    
    @staticmethod
    def build_config(config: dict) -> AttentionConfig:
        config = module_has_attr(config, "positional_embedding",
                                 primary_module=positional_embeddings,
                                 fallback_module=nn)
        
        return AttentionConfig(**config)