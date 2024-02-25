import torch
import torch.nn as nn

class AbsolutePE(nn.Module):
    # For absolute positional embedding, half of the embeddings comes from a
    # sin wave and the other half comes from a cos wave.
    
    def __init__(self, context_length: int = 1024, embedding_dim: int = 512):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.context_length = context_length

        if embedding_dim % 2 != 0:
            embedding_dim += 1
        d_model = embedding_dim//2
        
        # This is the scale used in Attention Is All You Need paper
        # (C/2)
        scale = 1 / 10_000**(2/d_model)
        freqs = torch.ones(d_model) * scale
        
        # Since we are going to want a positional embedding table we repeat
        # this to get the following shape
        # (C/2) -> (T, C/2)
        freqs = freqs ** torch.arange(0, d_model, 1).repeat(context_length, 1)
        
        # (T, 1)
        pos = torch.arange(0, context_length, 1).unsqueeze(-1)

        # (T, 1) * (T, C/2) -> (T, C/2) -> (T*C/2)
        sin = torch.sin(pos*freqs).view(-1)
        cos = torch.cos(pos*freqs).view(-1)
        
        # (2, T*C/2)
        sinusoids = torch.stack((sin, cos))
        # (2, T*C/2) -> (T*C/2, 2) -> (T*C) -> (T, C)
        self.positional_embeddings = sinusoids.t().contiguous().view(context_length, embedding_dim)
        self.positional_embeddings = nn.Parameter(self.positional_embeddings)

    def forward(self, x):
        out = None
        
        for x_batch in x:
            start, *_, end = x_batch
            embedding = self.positional_embeddings[start:end+1, :].unsqueeze(0)
            if out is None:
                out = embedding
            else:
                out = torch.cat((out, embedding), dim=0)
        
        return out