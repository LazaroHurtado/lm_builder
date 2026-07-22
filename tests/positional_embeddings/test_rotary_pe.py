import pytest
import torch
from torch.nn import functional as F
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
)

from lm_builder.attention import AttentionConfig, CausalMultiHeadAttention
from lm_builder.positional_embeddings.rotary_pe import RotaryPE

NUM_HEAD = 4
HEAD_DIM = 64
BASE = 10000.0
SEQ_LEN = 10


@pytest.fixture(autouse=True)
def clear_rope_instances():
    RotaryPE.KEY_TO_INSTANCE.clear()
    yield
    RotaryPE.KEY_TO_INSTANCE.clear()


@pytest.fixture
def rope():
    return RotaryPE(HEAD_DIM, SEQ_LEN, BASE)


@pytest.fixture
def hf_rope():
    config = LlamaConfig(
        max_position_embeddings=HEAD_DIM, head_dim=HEAD_DIM, rope_theta=BASE
    )
    return LlamaRotaryEmbedding(config, device="cpu")


def test_initialization_is_lazy(rope: RotaryPE):
    assert not isinstance(rope, torch.nn.Module)
    assert rope.embedding_dim == 64
    assert rope.base == 10000.0
    assert rope.inv_freq is None
    assert rope.cos_cached is None
    assert rope.sin_cached is None


def test_instances_are_reused_by_configuration():
    rope = RotaryPE(HEAD_DIM, SEQ_LEN, BASE)

    assert RotaryPE(HEAD_DIM, SEQ_LEN, BASE) is rope
    assert RotaryPE(HEAD_DIM + 2, SEQ_LEN, BASE) is not rope
    assert RotaryPE(HEAD_DIM, SEQ_LEN + 1, BASE) is not rope
    assert RotaryPE(HEAD_DIM, SEQ_LEN, BASE + 1) is not rope


def test_odd_embedding_dim_adjustment():
    rope = RotaryPE(63, SEQ_LEN, BASE)

    assert rope.embedding_dim == 64
    assert rope.inv_freq is None

    rope(
        torch.randn(1, NUM_HEAD, SEQ_LEN, rope.embedding_dim),
        torch.randn(1, NUM_HEAD, SEQ_LEN, rope.embedding_dim),
    )
    assert rope.inv_freq.shape == (32,)


def test_tables_are_generated_on_first_query_device_and_dtype(rope: RotaryPE):
    q = torch.randn(2, NUM_HEAD, SEQ_LEN, HEAD_DIM, dtype=torch.float16)
    k = torch.randn_like(q)

    rope(q, k)

    assert rope.inv_freq.device == q.device
    assert rope.inv_freq.dtype == torch.float32
    assert rope.cos_cached.device == q.device
    assert rope.cos_cached.dtype == q.dtype
    assert rope.sin_cached.device == q.device
    assert rope.sin_cached.dtype == q.dtype


def test_attention_layers_share_rope_tables(monkeypatch):
    config = AttentionConfig(
        context_length=SEQ_LEN,
        embedding_dimension=HEAD_DIM,
        num_heads=NUM_HEAD,
        attention_type=CausalMultiHeadAttention,
        positional_embedding=RotaryPE,
    )
    first = CausalMultiHeadAttention(config)
    second = CausalMultiHeadAttention(config)

    assert first.pos_emb is second.pos_emb
    assert "pos_emb" not in dict(first.named_modules())

    first(torch.randn(2, SEQ_LEN, HEAD_DIM))
    cos_cached = first.pos_emb.cos_cached
    sin_cached = first.pos_emb.sin_cached
    third = CausalMultiHeadAttention(config)

    assert third.pos_emb is first.pos_emb
    assert third.pos_emb.cos_cached is cos_cached
    assert third.pos_emb.sin_cached is sin_cached

    def fail_if_tables_are_recalculated(*args, **kwargs):
        pytest.fail("In-range position IDs should use the cached tables.")

    monkeypatch.setattr(
        first.pos_emb,
        "_get_cos_sin_embeddings",
        fail_if_tables_are_recalculated,
    )
    third(
        torch.randn(2, SEQ_LEN, HEAD_DIM),
        position_ids=torch.arange(SEQ_LEN).repeat(2, 1),
    )

    assert second.pos_emb.cos_cached is cos_cached
    assert second.pos_emb.sin_cached is sin_cached


def test_meta_queries_are_rejected(rope: RotaryPE):
    query = torch.empty(1, NUM_HEAD, 2, HEAD_DIM, device="meta")

    with pytest.raises(AssertionError, match="does not support meta"):
        rope(query, query)


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
    q = torch.randn(1, NUM_HEAD, SEQ_LEN, HEAD_DIM)
    rope(q, q)

    assert torch.allclose(rope.inv_freq, hf_rope.inv_freq, atol=1e-5)


def test_rope(rope: RotaryPE, hf_rope: LlamaRotaryEmbedding):
    """Test that the forward method produces similar results to Hugging Face's implementation."""
    batch_size = 2

    # Create dummy Q and K tensors
    q = torch.randn(batch_size, NUM_HEAD, SEQ_LEN, HEAD_DIM)
    k = torch.randn(batch_size, NUM_HEAD, SEQ_LEN, HEAD_DIM)
    # Position IDs
    position_ids = torch.arange(SEQ_LEN).unsqueeze(0).repeat(batch_size, 1)

    # Apply custom RotaryPE
    q_rotary, k_rotary = rope(q, k, unsqueeze_dim=1)

    # Apply Hugging Face's Rotary Embedding
    hf_cos, hf_sin = hf_rope(q, position_ids)
    q_hf_rotary, k_hf_rotary = apply_rotary_pos_emb(
        q, k, hf_cos, hf_sin, unsqueeze_dim=1
    )

    assert torch.allclose(q_rotary, q_hf_rotary, atol=1e-5)
    assert torch.allclose(k_rotary, k_hf_rotary, atol=1e-5)


@pytest.mark.parametrize(
    "position_ids",
    [
        torch.tensor([[0, 3, 5, 7], [1, 2, 4, 6]]),
        torch.tensor([[0, 3, 10, 12], [1, 2, 4, 11]]),
    ],
)
def test_explicit_position_ids_match_hugging_face(
    rope: RotaryPE,
    hf_rope: LlamaRotaryEmbedding,
    position_ids,
):
    q = torch.randn(2, NUM_HEAD, 4, HEAD_DIM)
    k = torch.randn_like(q)

    q_rotary, k_rotary = rope(q, k, position_ids=position_ids)
    hf_cos, hf_sin = hf_rope(q, position_ids)
    q_hf_rotary, k_hf_rotary = apply_rotary_pos_emb(
        q,
        k,
        hf_cos,
        hf_sin,
        unsqueeze_dim=1,
    )

    assert torch.allclose(q_rotary, q_hf_rotary, atol=1e-5)
    assert torch.allclose(k_rotary, k_hf_rotary, atol=1e-5)


def test_cached_decode_matches_full_rope_across_context_overflow():
    rope = RotaryPE(HEAD_DIM, 4, BASE)
    q = torch.randn(2, NUM_HEAD, 6, HEAD_DIM)
    k = torch.randn_like(q)
    position_ids = torch.arange(6).repeat(2, 1)

    full_q, full_k = rope(q, k, position_ids=position_ids)
    cached_q, cached_k = zip(
        *[
            rope(
                q[:, :, index : index + 1],
                k[:, :, index : index + 1],
                position_ids=position_ids[:, index : index + 1],
            )
            for index in range(position_ids.size(1))
        ]
    )

    torch.testing.assert_close(torch.cat(cached_q, dim=2), full_q)
    torch.testing.assert_close(torch.cat(cached_k, dim=2), full_k)


@pytest.mark.parametrize("use_scaled_dot_product_attention", [True, False])
def test_rope_preserves_half_precision_with_position_ids(
    use_scaled_dot_product_attention,
):
    attention = CausalMultiHeadAttention(
        AttentionConfig(
            context_length=SEQ_LEN,
            embedding_dimension=HEAD_DIM,
            num_heads=NUM_HEAD,
            attention_type=CausalMultiHeadAttention,
            positional_embedding=RotaryPE,
        )
    ).half()
    attention.has_flash_attn = use_scaled_dot_product_attention and hasattr(
        F, "scaled_dot_product_attention"
    )
    inputs = torch.randn(2, SEQ_LEN, HEAD_DIM, dtype=torch.float16)
    attention_mask = torch.ones(2, SEQ_LEN, dtype=torch.bool)
    position_ids = torch.arange(SEQ_LEN).repeat(2, 1)

    output = attention(
        inputs,
        attention_mask=attention_mask,
        position_ids=position_ids,
    )

    assert output.dtype == torch.float16
    assert torch.isfinite(output).all()
