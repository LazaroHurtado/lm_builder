import torch
import torch.nn as nn
import torch.nn.functional as F

from .block import Block
from .transformer_config import TransformerConfig

class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        
        self.embedding_dim = config.attention_config.embedding_dimension
        self.context_length  = config.attention_config.context_length
        
        blocks = [Block(
            config.attention(config.attention_config),
            config.ffn(config.ffn_config))
            for _ in range(config.num_layers)]
        self.transformer = nn.ModuleDict(dict(
            wte = config.token_embedding(config.vocab_size, self.embedding_dim),
            wpe = config.positional_embedding(self.context_length, self.embedding_dim),
            blocks = nn.ModuleList(blocks),
            dropout = nn.Dropout(config.dropout),
            ln_f = nn.LayerNorm(self.embedding_dim)
        ))
        # In reality this is just the wte weights but transposed so we can map
        # from embedding to vocabulary
        self.lm_head = nn.Linear(self.embedding_dim, config.vocab_size)

        self.config = config
    
    def forward(self, x, targets = None):
        _, T = x.size()
        assert T <= self.context_length
        
        pos = torch.arange(0, T, dtype=torch.long).unsqueeze(0)
        
        token_embeddings = self.transformer.wte(x) # (B, T, C)
        pos_embeddings = self.transformer.wpe(pos) # (1, T, C)
        
        x = self.transformer.dropout(token_embeddings+pos_embeddings) # (B, T, C)
        
        for block in self.transformer.blocks:
            x = block(x)
        x = self.transformer.ln_f(x)
        # (B, T, C) -> (B, T, V)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1).unsqueeze(0),
                                   targets.view(-1),
                                   ignore_index=-1)
        
        return logits, loss

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens, temperature=1.0):
        
        for _ in range(max_new_tokens):
            input_context_length = input_ids.shape[-1]
            if input_context_length > self.context_length:
                input_ids = input_ids[:, -self.context_length:]
            
            logits, _ = self(input_ids)
            logits = logits[:, -1, :] / temperature
            
            if self.config.top_k is not None:
                v, _ = torch.topk(logits, self.config.top_k)
                logits[logits < v[:, [-1]]] = float('-inf')
            
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            
            input_ids = torch.cat((input_ids, next_id), dim=1)
        
        return input_ids
