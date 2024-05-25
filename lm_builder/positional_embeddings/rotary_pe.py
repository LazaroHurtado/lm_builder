import torch
import torch.nn as nn

class RotaryPE(nn.Module):
    # For rotary positional embedding, we take chunks of two from the
    # token embeddings and apply a rotation.
    
    def __init__(self, context_length: int, embedding_dim: int, base: float):
        super().__init__()

        if embedding_dim % 2 != 0:
            embedding_dim += 1

        self.embedding_dim = embedding_dim
        self.context_length = context_length
        self.base = base
        
        self._generate_positional_embeddings()

    def _generate_positional_embeddings(self):
        # In the ReFormer paper, the positional embedding is applied to
        # Q and K matrices in the attention layer. It does so by taking
        # chunks of two from the token embeddings and applying a rotation
        # matrix

        # Since we are taking chunks of two from the token embeddings, we
        # start with half the size of the embedding dimension and then repeat
        # it twice to match the embedding dimension. In the ReFormer paper,
        # this is called the thetas but we will refer to it as the inverse
        # frequency, inv_freq
        # (C)
        power = torch.arange(0, self.embedding_dim, step=2) / self.embedding_dim
        inv_freq = 1 / (self.base**power)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # For each inv_freq, we want to multiply it by the position of the token.
        # We call this m to match the paper's notation
        # (T)
        m = torch.arange(0, self.context_length)
        
        # (T) X (C) -> (T, C) -> (T, 2*C)
        angles = torch.outer(m, inv_freq)
        angles = torch.cat((angles, angles), dim=-1)

        # (1, T, C)
        sin = torch.sin(angles).unsqueeze(0)
        cos = torch.cos(angles).unsqueeze(0)

        self.register_buffer("_sin_cached", sin, persistent=False)
        self.register_buffer("_cos_cached", cos, persistent=False)

    def interleave(self, x: torch.Tensor, shape: torch.Size):
        return x.t().contiguous().view(*shape)

    def forward(self, x: torch.Tensor, unsqueeze_dim=1):
        T = x.size(dim=-2)
        # Because we are applying a rotation using the sin and cos waves, we
        # have to break x up into its even and negative odd pairs to make the
        # computation easier. We do negative odd pairs because our sin tensor
        # is all positive.
        # Example:
        #   Assume T=1 and C=4 and x is the following tensor
        #   [[0.4, 0.2, 0.8, 0.3]]
        #   recall that for RoPE we take chunks of two and then apply a rotation, so
        #   the first rotation is:
        #   cos(a), -sin(a),  @  0.4,  =  0.4cos(a) - 0.2sin(a)  = 0.4cos(a) - 0.2sin(a)
        #   sin(a),  cos(a)      0.2      0.4sin(a) + 0.2cos(a)    0.2cos(a) + 0.4sin(a)
        #   If we were to expand this to a larger input tensor, we would
        #   see that cos(a) gets multiplied to all values in the tensor
        #   and -sin(a) gets multiplied only to odd indices while sin(a) gets
        #   multiplied only to even indices.
        # (B, T, C/2)
        neg_odds, pos_evens = -x[..., 1::2], x[..., ::2]
        # (B, T, C/2) -> (2, B*T*C/2)
        y = torch.stack([neg_odds.reshape(-1), pos_evens.reshape(-1)])
        # (2, B*T*C/2) -> (B, T, C)
        y = self.interleave(y, x.size())

        cos = self._cos_cached[:, :T, :].unsqueeze(unsqueeze_dim)
        sin = self._sin_cached[:, :T, :].unsqueeze(unsqueeze_dim)
        return x * cos + y * sin