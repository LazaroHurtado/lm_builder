import torch
import torch.nn as nn

class RotaryPE(nn.Module):
    # For rotary positional embedding, we take chunks of two from the
    # token embeddings and apply a rotation.
    
    def __init__(self, context_length: int = 1024, embedding_dim: int = 512):
        super().__init__()

        if embedding_dim % 2 != 0:
            embedding_dim += 1

        self.embedding_dim = embedding_dim
        self.context_length = context_length
        
        sin_pe, cos_pe = self._generate_positional_embeddings()
        
        self.register_buffer("sin_weight", sin_pe)
        self.register_buffer("cos_weight", cos_pe)

    def _generate_positional_embeddings(self):
        # In the ReFormer paper, the positional embedding is applied to
        # Q and K matrices in the attention layer. It does so by taking
        # chunks of two from the token embeddings and applying a rotation
        # matrix

        # Since we are taking chunks of two from the token embeddings, we
        # build the thetas to be half the size of the embedding dimension
        # and then repeat it twice to match the embedding dimension. We call
        # this theta to match the paper's notation
        # (1, C)
        power = (2*torch.arange(0, self.embedding_dim, step=2)) / self.embedding_dim
        thetas = 1 / (10_000**power).repeat_interleave(2).unsqueeze(0)
        
        # For each theta, we want to multiply it by the position of the token.
        # We call this m to match the paper's notation
        # (T, 1)
        m = torch.arange(0, self.context_length).unsqueeze(-1)
        
        # (T, 1) * (1, C) -> (T, C)
        wavelengths = m*thetas

        # (1, T, C)
        sin = torch.sin(wavelengths).unsqueeze(0)
        cos = torch.cos(wavelengths).unsqueeze(0)

        return (sin, cos)

    def interleave(self, x: torch.Tensor, shape: torch.Size):
        return x.t().contiguous().view(*shape)

    def forward(self, x: torch.Tensor):
        _, T, C = x.size()
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

        return x * self.cos_weight[:, :T, :C] + y * self.sin_weight[:, :T, :C]