import torch
from torch import nn
from torch.nn import functional as F

from .block import Block
from .config import TransformerConfig


class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()

        self.embedding_dim = config.embedding_dimension
        self.context_length = config.context_length

        blocks = [
            Block(
                attention_config=attention_config,
                ffn_config=config.ffn_config.clone(),
            )
            for attention_config in config.attention_config
        ]

        self.wte = config.token_embedding(config.vocab_size, self.embedding_dim)
        self.wpe = None
        self.blocks = nn.ModuleList(blocks)
        self.dropout = nn.Dropout(config.dropout)
        self.norm = config.norm.build(self.embedding_dim)

        if config.positional_embedding is not None:
            positional_embedding = config.positional_embedding(
                self.context_length, self.embedding_dim, config.inv_freq
            )
            self.wpe = positional_embedding

        self.lm_head = nn.Linear(
            self.embedding_dim, config.vocab_size, bias=config.bias
        )

        self.config = config
        if config.tie_word_embeddings:
            self._tie_word_embeddings()
            # Sometimes we build the model on meta device to save space,
            # so to not double the lm_head and wpe parameters we need to tie
            # them again after loading the state dict
            self.register_load_state_dict_post_hook(self._tie_word_embeddings)

    def _tie_word_embeddings(self, *args, **kwargs):  # pylint: disable=unused-argument
        self.lm_head.weight = self.wte.weight

    def _get_cached_sequence_length(self, kv_caches):
        if kv_caches is None:
            return 0
        if len(kv_caches) != len(self.blocks):
            raise ValueError("A KV cache is required for each transformer block.")

        cache_lengths = {cache.sequence_length for cache in kv_caches}
        if len(cache_lengths) > 1:
            raise ValueError("All transformer KV caches must have equal lengths.")

        return cache_lengths.pop()

    def _get_cached_tokens_seen(self, kv_caches):
        if kv_caches is None:
            return 0

        token_counts = {cache.tokens_seen for cache in kv_caches}
        if len(token_counts) > 1:
            raise ValueError("All transformer KV caches must have equal token counts.")

        return token_counts.pop()

    def _prepare_position_ids(
        self,
        x,
        position_ids,
        attention_mask,
        cached_tokens_seen,
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
                cached_tokens_seen,
                cached_tokens_seen + sequence_length,
                device=x.device,
                dtype=torch.long,
            ).expand(batch_size, -1)

        return None

    def forward(  # pylint: disable=too-many-locals
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
        cached_tokens_seen = self._get_cached_tokens_seen(_kv_caches)
        key_sequence_length = min(
            cached_sequence_length + T,
            self.context_length,
        )

        if attention_mask is not None:
            if (
                attention_mask.ndim != 2
                or attention_mask.size(0) != B
                or attention_mask.size(1) < key_sequence_length
            ):
                raise ValueError(
                    "Attention mask must contain the complete key sequence shape."
                )
            attention_mask = attention_mask.to(device=x.device, dtype=bool)

        position_ids = self._prepare_position_ids(
            x,
            position_ids,
            attention_mask,
            cached_tokens_seen,
            _kv_caches,
        )

        # Token embedding layer
        x = self.wte(x)  # (B, T, C)
        if self.wpe is not None:
            x = self.wpe(x, position_ids=position_ids)

        x = self.dropout(x)  # (B, T, C)

        routing_losses = []
        for layer_index, block in enumerate(self.blocks):
            kv_cache = None if _kv_caches is None else _kv_caches[layer_index]
            x, routing_loss = block(
                x,
                attention_mask=attention_mask,
                position_ids=position_ids,
                kv_cache=kv_cache,
            )
            if routing_loss is not None:
                routing_losses.append(routing_loss)
        x = self.norm(x)
        # (B, T, C) -> (B, T, V)
        logits = self.lm_head(x)

        cross_entropy_loss = None
        if targets is not None:
            cross_entropy_loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-1,
            )

        routing_loss = None
        if routing_losses:
            routing_loss = torch.stack(routing_losses).mean()

        return logits, cross_entropy_loss, routing_loss
