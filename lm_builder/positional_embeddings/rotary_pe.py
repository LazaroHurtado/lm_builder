import torch

from einops import rearrange
from torch import nn


class RotaryPE(nn.Module):
    # For rotary positional embedding, we take chunks of two from the
    # token embeddings and apply a rotation.

    def __init__(self, context_length: int, dim: int, freq: float):
        super().__init__()

        if dim % 2 != 0:
            dim += 1

        self.dim = dim
        self.context_length = context_length
        self.freq = freq

        self._generate_positional_embeddings()

    def _generate_positional_embeddings(self):
        # In the ReFormer paper, the positional embedding is applied to
        # Q and K matrices in the attention layer. It does so by taking
        # chunks of two from the token embeddings and applying a rotation
        # matrix

        # In the ReFormer paper, each pair of dimensions in the embedding
        # tensor is treated as a point on the complex plane, which is why
        # we are stepping by 2. The paper also calls this the thetas but
        # we will refer to it as the inverse frequency, inv_freq
        # (C/2)
        power = (2 * torch.arange(0, self.dim, step=2)) / self.dim
        inv_freq = 1 / (self.freq**power)

        # For each inv_freq, we want to multiply it by the position of the token.
        # We call this m to match the paper's notation
        # (T)
        m = torch.arange(0, self.context_length)

        # (T) X (C/2) -> (T, C/2)
        angles = torch.outer(m, inv_freq)

        # One way to represent a rotation is through Euler's formula, e^(i*theta),
        # which is why we are using polar, so we can get a complex tensor of the form
        # e^(i*m*inv_freq)
        # (T, C/2)
        rotations = torch.polar(torch.ones_like(angles), angles)

        self.register_buffer("_rotations_cached", rotations, persistent=False)

    def forward(self, x: torch.Tensor):
        T = x.size(dim=1)  # pylint: disable=invalid-name
        C = x.size(dim=-1)  # pylint: disable=invalid-name

        # As previously mentioned, the paper represents each pair of dimensions
        # as a point on the complex plane. Thus, we rearrange the tensor to get
        # pairs of two.
        # (B, T, num_heads, head_dim/2, 2)
        x_pairs = rearrange(x, "... (head_dim r) -> ... head_dim r", r=2)

        # Once we have the pairs we can convert them to a complex representation.
        # Example:
        #   Let x = [[1.2, 0.8], [0.2, 0.7]], then the complex representation would be
        #   c = [1.2+0.8i, 0.2+0.7i]
        # (B, T, num_heads, head_dim/2)
        x_as_complex = torch.view_as_complex(x_pairs)

        # We change the shape of our rotations tensor to match x
        # (T, C/2) -> (1, T, C/2) -> (1, T, 1, C/2)
        rotations = self._rotations_cached[:T, :C].unsqueeze(0).unsqueeze(2)

        # (B, T, num_heads, head_dim/2) * (1, T, 1, C/2) -> (B, T, num_heads, head_dim/2, 2)
        out = torch.view_as_real(x_as_complex * rotations)

        # (B, T, num_heads, head_dim)
        return out.flatten(3)
