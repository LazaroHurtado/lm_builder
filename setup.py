#!/usr/bin/env python3

from pathlib import Path
from setuptools import setup

directory = Path(__file__).resolve().parent
with open(directory / "README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="lm_builder",
    description="Python framework for building language models",
    author="Lazaro Hurtado",
    version="0.0.1",
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=[
        "lm_builder",
        "lm_builder.attention",
        "lm_builder.ffn",
        "lm_builder.positional_embeddings",
        "lm_builder.transformer",
    ],
    install_requires=[
        "accelerate",
        "einops",
        "python-dotenv",
        "torch",
        "tiktoken",
        "transformers",
    ],
    extra_require={"linting": ["pylint"]},
    python_requires=">=3.8",
)
