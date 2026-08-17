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

        qk_positional_embedding_config = config.attention_config.qk_positional_embedding
        self.qk_positional_embedding = None
        if qk_positional_embedding_config is not None:
            self.qk_positional_embedding = qk_positional_embedding_config.build(
                config.attention_config.layers[0].head_dim,
                self.context_length,
            )

        blocks = [
            Block(
                attention_config=attention_config,
                ffn_config=config.ffn_config.clone(),
                qk_positional_embedding=self.qk_positional_embedding,
            )
            for attention_config in config.attention_config.layers
        ]

        self.wte = config.token_embedding(config.vocab_size, self.embedding_dim)
        self.wpe = (
            config.positional_embedding(
                self.context_length, self.embedding_dim, config.inv_freq
            )
            if config.positional_embedding is not None
            else None
        )
        self.blocks = nn.ModuleList(blocks)
        self.dropout = nn.Dropout(config.dropout)
        self.norm = config.norm.build(self.embedding_dim)

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

    @staticmethod
    def _prepare_position_ids(
        x,
        position_ids,
        attention_mask,
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
        return torch.arange(
            sequence_length,
            device=x.device,
            dtype=torch.long,
        ).expand(batch_size, -1)

    def forward(  # pylint: disable=too-many-locals
        self,
        x,
        targets=None,
        attention_mask=None,
        position_ids=None,
        cache_position=None,
        *,
        logits_to_keep=0,
        _kv_caches=None,
    ):
        B, T = x.size()  # pylint: disable=invalid-name
        assert T <= self.context_length
        if (
            not isinstance(logits_to_keep, int)
            or isinstance(logits_to_keep, bool)
            or logits_to_keep < 0
        ):
            raise ValueError("logits_to_keep must be a non-negative integer.")
        if targets is not None and logits_to_keep:
            raise ValueError("logits_to_keep must be 0 when targets are provided.")

        if _kv_caches is not None:
            if len(_kv_caches) != len(self.blocks):
                raise ValueError("A KV cache is required for each transformer block.")
            if position_ids is None:
                raise ValueError("Position IDs are required when using a KV cache.")
            if cache_position is None:
                raise ValueError("Cache positions are required when using a KV cache.")
            if cache_position.ndim != 1 or cache_position.numel() != T:
                raise ValueError(
                    "Cache positions must match the input sequence length."
                )
            cache_position = cache_position.to(device=x.device, dtype=torch.long)

        if attention_mask is not None:
            if (
                attention_mask.ndim != 2
                or attention_mask.size(0) != B
                or attention_mask.size(1) < T
            ):
                raise ValueError("Attention mask must cover the input sequence.")
            attention_mask = attention_mask.to(device=x.device, dtype=bool)
            if _kv_caches is not None:
                attention_mask = attention_mask[:, -T:]

        position_ids = self._prepare_position_ids(
            x,
            position_ids,
            attention_mask,
        )

        # Token embedding layer
        x = self.wte(x)  # (B, T, C)
        if self.wpe is not None:
            x = self.wpe(x, position_ids=position_ids)

        x = self.dropout(x)  # (B, T, C)

        qk_position_data = None
        if self.qk_positional_embedding is not None:
            qk_position_data = self.qk_positional_embedding.prepare(
                x,
                position_ids,
            )

        routing_losses = []
        for layer_index, block in enumerate(self.blocks):
            kv_cache = None if _kv_caches is None else _kv_caches[layer_index]
            x, routing_loss = block(
                x,
                attention_mask=attention_mask,
                qk_position_data=qk_position_data,
                kv_cache=kv_cache,
                cache_position=cache_position,
            )
            if routing_loss is not None:
                routing_losses.append(routing_loss)
        x = self.norm(x)
        if logits_to_keep:
            x = x[:, -logits_to_keep:, :]
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
