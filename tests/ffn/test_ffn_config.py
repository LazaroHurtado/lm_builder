from lm_builder.ffn import FeedForward, FeedForwardConfig


def test_from_yml_uses_top_level_embedding_dimension(tmp_path):
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

    config = FeedForwardConfig.from_yml(config_path)

    assert config.embedding_dimension == 16
    assert config.ffn_type is FeedForward
