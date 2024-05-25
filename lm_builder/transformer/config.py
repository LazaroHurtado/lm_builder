from __future__ import annotations

import torch.nn as nn
import yaml

from .. import attention
from .. import ffn
from .. import positional_embeddings
from ..utils import module_has_attr

from dataclasses import dataclass
from typing import Optional

@dataclass
class TransformerConfig():
    attention: attention.Attention
    attention_config: attention.AttentionConfig
    ffn: ffn.FeedForward
    ffn_config: ffn.FeedForwardConfig
    vocab_size: int
    num_layers: int
    token_embedding: nn.Module
    bias: bool = False
    attn_norm: Optional[nn.Module] = None
    mlp_norm: Optional[nn.Module] = None
    transformer_norm: Optional[nn.Module] = None
    positional_embedding: Optional[nn.Module] = None
    inv_freq: Optional[float] = 10_000.0
    dropout: Optional[float] = 0.0
    top_k: Optional[int] = 2

    @staticmethod
    def from_yml(file: str) -> TransformerConfig:
        with open(file, 'r') as f:
            config = yaml.safe_load(f)
            
            return TransformerConfig.build_config(config)
    
    @staticmethod
    def build_config(config: dict) -> TransformerConfig:
        config = module_has_attr(config, "attention",
                                 primary_module=attention,
                                 fallback_module=nn)
            
        config = module_has_attr(config, "ffn",
                                 primary_module=ffn,
                                 fallback_module=nn)

        config = module_has_attr(config, "positional_embedding",
                                 primary_module=positional_embeddings,
                                 fallback_module=nn)
        
        config = module_has_attr(config, "token_embedding", nn)
        config = module_has_attr(config, "transformer_norm", nn)
        config = module_has_attr(config, "attn_norm", nn)
        config = module_has_attr(config, "mlp_norm", nn)
        
        config["attention_config"] = attention.AttentionConfig.build_config(config["attention_config"])
        config["ffn_config"] = ffn.FeedForwardConfig.build_config(config["ffn_config"])
        
        return TransformerConfig(**config)