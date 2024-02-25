
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
    def attn_mask(cls): ...