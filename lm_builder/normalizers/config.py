from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Type

from torch import nn

from ..utils import module_has_attr


@dataclass
class NormalizerConfig:
    normalizer_type: Type[nn.Module] = nn.LayerNorm
    kwargs: Dict[str, object] = field(default_factory=lambda: {"bias": False})

    def build(self, dimension: int) -> nn.Module:
        return self.normalizer_type(dimension, **self.kwargs)

    def clone(self) -> NormalizerConfig:
        return NormalizerConfig(
            normalizer_type=self.normalizer_type,
            kwargs=dict(self.kwargs),
        )

    @staticmethod
    def build_config(config=None) -> NormalizerConfig:
        if config is None:
            return NormalizerConfig()
        if isinstance(config, NormalizerConfig):
            return config
        if not isinstance(config, dict):
            raise TypeError("norm must be a mapping.")

        config = dict(config)

        # Imported here to avoid importing the package while it is still
        # initializing this config module.
        from lm_builder import normalizers  # pylint: disable=import-outside-toplevel

        config = module_has_attr(
            config,
            "type",
            primary_module=normalizers,
            fallback_module=nn,
        )
        normalizer_type = config.pop("type", nn.LayerNorm)
        config.setdefault("bias", False)

        return NormalizerConfig(
            normalizer_type=normalizer_type,
            kwargs=config,
        )
