import torch

from torch import nn


class RMSNorm(nn.Module):

    def __init__(self, dim: int, eps: float = 1e-6, **kwargs):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self._norm(x) * self.weight
