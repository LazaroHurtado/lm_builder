import pytest

from lm_builder.ffn import FeedForward, FeedForwardConfig
from lm_builder.utils import load_yml


def test_build_config_uses_top_level_embedding_dimension(tmp_path):
    config_path = tmp_path / "model.yml"
    config_path.write_text(
        """
embedding_dimension: 16
ffn_config:
  type: FeedForward
  intermediate_dimension: 32
""",
        encoding="utf-8",
    )

    raw_config = load_yml(config_path)
    config = FeedForwardConfig.build_config(
        raw_config["ffn_config"],
        raw_config["embedding_dimension"],
    )

    assert config.embedding_dimension == 16
    assert config.ffn_type is FeedForward


def test_build_config_requires_type():
    with pytest.raises(ValueError, match="ffn_config.type is required"):
        FeedForwardConfig.build_config({"intermediate_dimension": 32}, 16)
