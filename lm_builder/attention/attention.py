import torch

from abc import abstractmethod
from .config import AttentionConfig

class Attention():
    @property
    def config(self) -> AttentionConfig:
        return self._config
    
    @config.setter
    def config(self, new_config):
        return

    @abstractmethod
    def get_qkv(self, x: torch.Tensor): ...

    @abstractmethod
    def get_heads(self,
                  query: torch.Tensor,
                  key: torch.Tensor,
                  value: torch.Tensor): ...

    @abstractmethod
    def attention(self,
                  query: torch.Tensor,
                  key: torch.Tensor,
                  value: torch.Tensor): ...