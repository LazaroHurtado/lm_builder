from __future__ import annotations

import torch.nn as nn
import yaml

from dataclasses import dataclass
from typing import Optional
from ..utils import module_has_attr

@dataclass
class FeedForwardConfig():
    embedding_dimension: int
    intermediate_dimension: int
    activation_fn: Optional[nn.Module] = nn.GELU()
    dropout: Optional[float] = 0.0
    num_experts: Optional[int] = 0
    top_k: Optional[int] = 0

    @staticmethod
    def from_yml(file: str) -> FeedForwardConfig:
        with open(file, 'r') as f:
            config = yaml.safe_load(f)

            if "feed_forward_config" in config:
                config = config["feed_forward_config"]
            elif "ffn_config" in config:
                config = config["ffn_config"]
            
            config = module_has_attr(config, "activation_fn", nn)
            
            return FeedForwardConfig(**config)