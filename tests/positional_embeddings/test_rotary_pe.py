import pytest
import torch
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
)

from lm_builder.positional_embeddings.rotary_pe import RotaryPE

NUM_HEAD = 4
HEAD_DIM = 64
BASE = 10000.0


@pytest.fixture
def rope():
    return RotaryPE(HEAD_DIM, BASE)


@pytest.fixture
def hf_rope():
    config = LlamaConfig(
        max_position_embeddings=HEAD_DIM, head_dim=HEAD_DIM, rope_theta=BASE
    )
    return LlamaRotaryEmbedding(config, device="cpu")


def test_initialization(rope: RotaryPE):
    """Test that the module initializes with correct attributes and buffer shapes."""
    assert rope.embedding_dim == 64
    assert rope.base == 10000.0
    assert hasattr(rope, "inv_freq")
    # inv_freq size should be embedding_dim / 2
    assert rope.inv_freq.shape == (32,)


def test_odd_embedding_dim_adjustment():
    """Test that odd embedding dimensions are adjusted to be even."""
    rope = RotaryPE(63, 10000.0)
    assert rope.embedding_dim == 64
    assert rope.inv_freq.shape == (32,)


def test_rotate_half(rope: RotaryPE):
    """
    Test the rotate_half helper function.
    Logic: x = [x1, x2] -> [-x2, x1]
    """
    # Shape: (1, 4) for simplicity. x1=[1,2], x2=[3,4]
    x = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])

    # Expected: [-3.0, -4.0, 1.0, 2.0]
    expected = torch.tensor([[[-3.0, -4.0, 1.0, 2.0]]])

    rotated = rope.rotate_half(x)
    assert torch.allclose(rotated, expected)


def test_rope_inv_freq(rope: RotaryPE, hf_rope: LlamaRotaryEmbedding):
    """Test that the inv_freq buffer matches Hugging Face's implementation."""
    assert torch.allclose(rope.inv_freq, hf_rope.inv_freq, atol=1e-5)


def test_rope(rope: RotaryPE, hf_rope: LlamaRotaryEmbedding):
    """Test that the forward method produces similar results to Hugging Face's implementation."""
    seq_len = 10
    batch_size = 2

    # Create dummy Q and K tensors
    q = torch.randn(batch_size, NUM_HEAD, seq_len, HEAD_DIM)
    k = torch.randn(batch_size, NUM_HEAD, seq_len, HEAD_DIM)

    # Position IDs
    position_ids = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)

    # Apply custom RotaryPE
    q_rotary, k_rotary = rope(q, k, position_ids, unsqueeze_dim=1)

    # Apply Hugging Face's Rotary Embedding
    hf_cos, hf_sin = hf_rope(q, position_ids)
    q_hf_rotary, k_hf_rotary = apply_rotary_pos_emb(
        q, k, hf_cos, hf_sin, unsqueeze_dim=1
    )

    assert torch.allclose(q_rotary, q_hf_rotary, atol=1e-5)
    assert torch.allclose(k_rotary, k_hf_rotary, atol=1e-5)
