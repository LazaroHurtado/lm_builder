from abc import abstractmethod

import torch
from torch import nn


class Attention(nn.Module):
    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        attention_mask=None,
        qk_position_data=None,
        kv_cache=None,
    ): ...

    @abstractmethod
    def get_qkv(self, x: torch.Tensor): ...

    @abstractmethod
    def get_heads(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ): ...

    @abstractmethod
    def attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask=None,
    ): ...
