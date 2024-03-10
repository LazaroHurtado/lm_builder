import torch
import torch.nn as nn

class AbsolutePE(nn.Module):
    # For absolute positional embedding, half of the embeddings comes from a
    # sin wave and the other half comes from a cos wave.
    
    def __init__(self, context_length: int = 1024, embedding_dim: int = 512):
        super().__init__()

        if embedding_dim % 2 != 0:
            embedding_dim += 1

        self.embedding_dim = embedding_dim
        self.context_length = context_length
        
        self.register_buffer("weight", self._generate_positional_embeddings())

    def _generate_positional_embeddings(self):
        # In the Attention Is All You Need paper, the positional embedding table
        # is generated using sine for even index positions and cosine for odd positions.
        
        # We step by 2 so we can generate the sin and cos, which each takes half the
        # total embedding dimension, waves separately and then stack them together.
        power = (2*torch.arange(0, self.embedding_dim, step=2)) / self.embedding_dim
        
        # This is the scale used in the paper, 1/(10_000**(2i/d_model)) where i is the
        # dimension of the embedding
        # (1, C/2)
        scale = 1 / (10_000**power).unsqueeze(0)
        

        # The next step is to multiply the position by the scaling factor, pos/10_000**(2i/d_model)
        # (T, 1) * (1, C/2) -> (T, C/2)
        pos = torch.arange(0, self.context_length).unsqueeze(-1)
        wavelength = pos*scale

        # This will give us abosolute positional embedding for each position
        # and dimension of the embedding as described in the paper.
        # (T, C/2) -> (T*C/2)
        sin = torch.sin(wavelength).view(-1)
        cos = torch.cos(wavelength).view(-1)
        
        # (2, T*C/2)
        sinusoids = torch.stack((sin, cos))
        # (1, T, C)
        return self.interleave(sinusoids, (1, self.context_length, self.embedding_dim))

    def interleave(self, x: torch.Tensor, shape: torch.Size):
        # I will explain this through an example:
        # Example:
        #   Assume T=2 and C=3 and x is the following tensor
        #   [[0.0, 0.2, 0.4],   <- sin
        #    [1.0, 0.5, 0.1]]   <- cos
        #   then what we want is the following
        #   [[0.0, 1.0, 0.2],
        #    [0.5, 0.4, 0.1]]
        #   which interleaves the sin and cos values. We can get
        #   this behavior by first transposing the x tensor
        #   [[0.0, 1.0]
        #    [0.2, 0.5]
        #    [0.4, 0.1]]
        #   getting a contiguous view
        #   [0.0, 1.0, 0.2, 0.5, 0.4, 0.1]
        #   and then reshaping it to the desired shape we get the desired result
        #   [[0.0, 1.0, 0.2],
        #    [0.5, 0.4, 0.1]]
        # (2, T*C/2) -> (T*C/2, 2) -> (T*C) -> (T, C)
        return x.t().contiguous().view(*shape)

    def forward(self, x: torch.Tensor):
        _, T, C = x.size()
        return x + self.weight[:, :T, :C]