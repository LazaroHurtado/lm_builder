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

        attention_types = self._get_attention_types(config)
        blocks = [
            Block(config, attention_type=attention_type)
            for attention_type in attention_types
        ]

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

    @staticmethod
    def _get_attention_types(config):
        attention_ratio = config.attention_config.get_attention_ratio()
        if attention_ratio is None:
            return [config.attention] * config.num_layers

        attention_pattern = [
            attention_type
            for attention_type, layer_count in zip(config.attention, attention_ratio)
            for _ in range(layer_count)
        ]

        return [
            attention_pattern[layer_index % len(attention_pattern)]
            for layer_index in range(config.num_layers)
        ]

    def _get_cached_sequence_length(self, kv_caches):
        if kv_caches is None:
            return 0
        if len(kv_caches) != len(self.transformer.blocks):
            raise ValueError("A KV cache is required for each transformer block.")

        cache_lengths = {cache.sequence_length for cache in kv_caches}
        if len(cache_lengths) > 1:
            raise ValueError("All transformer KV caches must have equal lengths.")

        return cache_lengths.pop() if cache_lengths else 0

    def _prepare_position_ids(
        self,
        x,
        position_ids,
        attention_mask,
        cached_sequence_length,
        kv_caches,
    ):
        batch_size, sequence_length = x.size()
        expected_shape = (batch_size, sequence_length)

        if position_ids is not None:
            if position_ids.shape != expected_shape:
                raise ValueError("Position IDs must match the input IDs shape.")
            return position_ids.to(device=x.device, dtype=torch.long)

        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(dim=-1) - 1
            position_ids.masked_fill_(~attention_mask, 0)
            return position_ids[:, -sequence_length:]

        if kv_caches is not None:
            return torch.arange(
                cached_sequence_length,
                cached_sequence_length + sequence_length,
                device=x.device,
                dtype=torch.long,
            ).expand(batch_size, -1)

        return None

    def forward(
        self,
        x,
        targets=None,
        attention_mask=None,
        position_ids=None,
        *,
        _kv_caches=None,
    ):
        B, T = x.size()  # pylint: disable=invalid-name
        assert T <= self.context_length

        cached_sequence_length = self._get_cached_sequence_length(_kv_caches)
        key_sequence_length = cached_sequence_length + T
        if key_sequence_length > self.context_length:
            raise ValueError("Input and KV cache exceed the model context length.")

        expected_attention_shape = (B, key_sequence_length)
        if attention_mask is not None:
            if attention_mask.shape != expected_attention_shape:
                raise ValueError(
                    "Attention mask must match the complete key sequence shape."
                )
            attention_mask = attention_mask.to(device=x.device, dtype=bool)

        position_ids = self._prepare_position_ids(
            x,
            position_ids,
            attention_mask,
            cached_sequence_length,
            _kv_caches,
        )

        # Token embedding layer
        x = self.transformer.wte(x)  # (B, T, C)
        if "wpe" in self.transformer:
            # Positional embedding layer
            x = self.transformer.wpe(x, position_ids=position_ids)  # (1, T, C)

        x = self.transformer.dropout(x)  # (B, T, C)

        for layer_index, block in enumerate(self.transformer.blocks):
            kv_cache = None if _kv_caches is None else _kv_caches[layer_index]
            x = block(
                x,
                attention_mask=attention_mask,
                position_ids=position_ids,
                kv_cache=kv_cache,
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
