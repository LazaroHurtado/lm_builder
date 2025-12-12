from abc import abstractmethod

import torch
import torch.nn as nn


class Attention(nn.Module):
    @abstractmethod
    def get_qkv(self, x: torch.Tensor): ...

    @abstractmethod
    def get_heads(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ): ...

    @abstractmethod
    def attention(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ): ...
