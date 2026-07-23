import torch
from torch import nn


class AbsolutePE(nn.Module):
    def __init__(self, context_length: int, embedding_dim: int, base: float):
        super().__init__()

        if embedding_dim % 2 != 0:
            embedding_dim += 1

        self.embedding_dim = embedding_dim
        self.context_length = context_length
        self.base = base

        power = (
            2
            * torch.arange(
                0,
                self.embedding_dim,
                step=2,
                device="cpu",
                dtype=torch.float32,
            )
            / self.embedding_dim
        )
        inv_freq = 1 / (self.base**power)
        pos = torch.arange(
            self.context_length,
            device="cpu",
            dtype=torch.float32,
        )
        angles = torch.outer(pos, inv_freq)
        weight = torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(1)
        self.register_buffer("weight", weight, persistent=False)

    def forward(self, x: torch.Tensor, position_ids=None):
        _, T, C = x.size()
        
        if position_ids is None:
            positional_embedding = self.weight[None, :T, :C]
        else:
            position_ids = position_ids.to(device=x.device, dtype=torch.long)
            positional_embedding = self.weight[position_ids, :C]

        return x + positional_embedding.to(device=x.device, dtype=x.dtype)
