import torch

from .block import Block
from .config import TransformerConfig

from torch import nn
from torch.nn import functional as F


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
                self.context_length, self.embedding_dim, config.pos_emb_freq
            )

        self.transformer = nn.ModuleDict(transformer_modules)
        # In reality this is just the wte weights but transposed so we can map
        # from embedding to vocabulary
        self.lm_head = nn.Linear(
            self.embedding_dim, config.vocab_size, bias=config.bias
        )

        self.config = config

    def forward(self, x, targets=None):
        _, T = x.size()  # pylint: disable=invalid-name
        assert T <= self.context_length

        # Token embedding layer
        x = self.transformer.wte(x)  # (B, T, C)
        if "wpe" in self.transformer:
            # Positional embedding layer
            x = self.transformer.wpe(x)  # (1, T, C)

        x = self.transformer.dropout(x)  # (B, T, C)

        for block in self.transformer.blocks:
            x = block(x)
        x = self.transformer.norm(x)
        # (B, T, C) -> (B, T, V)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1).unsqueeze(0), targets.view(-1), ignore_index=-1
            )

        return logits, loss
