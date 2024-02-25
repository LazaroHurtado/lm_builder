from __future__ import annotations

import yaml

from dataclasses import dataclass
from typing import Optional

@dataclass
class AttentionConfig():
    context_length: int
    embedding_dimension: int
    num_heads: int
    with_kv_cache: Optional[bool] = False
    attn_dropout: Optional[float] = 0.0
    resid_dropout: Optional[float] = 0.0

    @staticmethod
    def from_yml(file: str) -> AttentionConfig:
        with open(file, 'r') as f:
            config = yaml.safe_load(f)
            if "attention_config" in config:
                config = config["attention_config"]
            return AttentionConfig(**config)
