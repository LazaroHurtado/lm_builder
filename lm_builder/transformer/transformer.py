import torch
from torch import nn
from torch.nn import functional as F

from .block import Block
from .config import TransformerConfig


class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()

        self.embedding_dim = config.attention_config.embedding_dimension
        self.context_length = config.attention_config.context_length

        blocks = [Block(config) for _ in range(config.num_layers)]

        transformer_modules = dict(
            wte=config.token_embedding(config.vocab_size, self.embedding_dim),
            blocks=nn.ModuleList(blocks),
            dropout=nn.Dropout(config.dropout),
            norm=config.norm(self.embedding_dim, bias=config.norm_bias),
        )

        if config.positional_embedding is not None:
            transformer_modules["wpe"] = config.positional_embedding(
                self.context_length, self.embedding_dim, config.inv_freq
            )

        self.transformer = nn.ModuleDict(transformer_modules)
        # In reality this is just the wte weights but transposed so we can map
        # from embedding to vocabulary
        self.lm_head = nn.Linear(
            self.embedding_dim, config.vocab_size, bias=config.bias
        )

        self.config = config

    def forward(self, x, targets=None, attention_mask=None, position_ids=None):
        B, T = x.size()  # pylint: disable=invalid-name
        assert T <= self.context_length

        expected_shape = (B, T)
        if attention_mask is not None:
            if attention_mask.shape != expected_shape:
                raise ValueError("Attention mask must match the input IDs shape.")
            attention_mask = attention_mask.to(device=x.device, dtype=bool)

        if position_ids is not None:
            if position_ids.shape != expected_shape:
                raise ValueError("Position IDs must match the input IDs shape.")
            position_ids = position_ids.to(device=x.device, dtype=torch.long)
        elif attention_mask is not None:
            position_ids = attention_mask.long().cumsum(dim=-1) - 1
            position_ids.masked_fill_(~attention_mask, 0)

        # Token embedding layer
        x = self.transformer.wte(x)  # (B, T, C)
        if "wpe" in self.transformer:
            # Positional embedding layer
            x = self.transformer.wpe(x, position_ids=position_ids)  # (1, T, C)

        x = self.transformer.dropout(x)  # (B, T, C)

        for block in self.transformer.blocks:
            x = block(
                x,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
        x = self.transformer.norm(x)
        # (B, T, C) -> (B, T, V)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-1,
            )

        return logits, loss
