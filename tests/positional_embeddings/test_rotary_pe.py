import pytest
import torch
from torch.nn import functional as F
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
)

from lm_builder.attention import (
    AttentionLayerConfig,
    CausalMultiHeadAttention,
)
from lm_builder.positional_embeddings.rotary_pe import RotaryPE

NUM_HEAD = 4
HEAD_DIM = 64
BASE = 10000.0
SEQ_LEN = 10
BATCH_SIZE = 2


@pytest.fixture
def rope():
    return RotaryPE(HEAD_DIM, SEQ_LEN, BASE)


@pytest.fixture
def hf_rope():
    config = LlamaConfig(
        max_position_embeddings=HEAD_DIM,
        head_dim=HEAD_DIM,
        rope_theta=BASE,
    )
    return LlamaRotaryEmbedding(config, device="cpu")


def test_initialization_has_no_mutable_position_cache(rope: RotaryPE):
    assert not isinstance(rope, torch.nn.Module)
    assert rope.embedding_dim == HEAD_DIM
    assert rope.context_len == SEQ_LEN
    assert rope.base == BASE
    assert rope.inv_freq.shape == (HEAD_DIM // 2,)
    assert rope.inv_freq.dtype == torch.float32
    assert not hasattr(rope, "cos_cached")
    assert not hasattr(rope, "sin_cached")


def test_instances_are_independent():
    first = RotaryPE(HEAD_DIM, SEQ_LEN, BASE)
    second = RotaryPE(HEAD_DIM, SEQ_LEN, BASE)

    assert first is not second


def test_inverse_frequencies_remain_materialized_during_meta_construction():
    with torch.device("meta"):
        rope = RotaryPE(HEAD_DIM, SEQ_LEN, BASE)

    assert rope.inv_freq.device.type == "cpu"


def test_odd_embedding_dim_adjustment():
    rope = RotaryPE(63, SEQ_LEN, BASE)

    assert rope.embedding_dim == 64
    assert rope.inv_freq.shape == (32,)


def test_prepare_matches_query_device_dtype_and_shape(rope: RotaryPE):
    x = torch.randn(2, NUM_HEAD, SEQ_LEN, HEAD_DIM, dtype=torch.float16)
    position_ids = torch.arange(SEQ_LEN).repeat(2, 1)

    cos, sin = rope.prepare(x, position_ids)

    assert cos.shape == (2, SEQ_LEN, HEAD_DIM)
    assert sin.shape == (2, SEQ_LEN, HEAD_DIM)
    assert cos.device == x.device
    assert sin.device == x.device
    assert cos.dtype == x.dtype
    assert sin.dtype == x.dtype


def test_apply_matches_qk_dtype_when_prepared_from_fp32_residuals(rope: RotaryPE):
    residuals = torch.randn(BATCH_SIZE, SEQ_LEN, NUM_HEAD * HEAD_DIM)
    position_ids = torch.arange(SEQ_LEN).expand(BATCH_SIZE, -1)
    q = torch.randn(
        BATCH_SIZE,
        NUM_HEAD,
        SEQ_LEN,
        HEAD_DIM,
        dtype=torch.bfloat16,
    )
    k = torch.randn_like(q)

    q_rotated, k_rotated = rope.apply_qk(
        q,
        k,
        rope.prepare(residuals, position_ids),
    )

    assert q_rotated.dtype == q.dtype
    assert k_rotated.dtype == k.dtype


def test_meta_queries_are_rejected(rope: RotaryPE):
    x = torch.empty(1, 2, HEAD_DIM, device="meta")
    position_ids = torch.empty(1, 2, dtype=torch.long, device="meta")

    with pytest.raises(AssertionError, match="does not support meta"):
        rope.prepare(x, position_ids)


def test_rotate_half(rope: RotaryPE):
    x = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
    expected = torch.tensor([[[-3.0, -4.0, 1.0, 2.0]]])

    rotated = rope.rotate_half(x)

    assert torch.allclose(rotated, expected)


def test_rope_inv_freq(rope: RotaryPE, hf_rope: LlamaRotaryEmbedding):
    assert torch.allclose(rope.inv_freq, hf_rope.inv_freq, atol=1e-5)


@pytest.mark.parametrize(
    "position_ids",
    [
        torch.arange(SEQ_LEN).repeat(2, 1),
        torch.tensor([[0, 3, 5, 7], [1, 2, 4, 6]]),
        torch.tensor([[0, 3, 10, 12], [1, 2, 4, 11]]),
    ],
)
def test_explicit_position_ids_match_hugging_face(
    rope: RotaryPE,
    hf_rope: LlamaRotaryEmbedding,
    position_ids,
):
    q = torch.randn(2, NUM_HEAD, position_ids.size(1), HEAD_DIM)
    k = torch.randn_like(q)

    position_data = rope.prepare(q, position_ids)
    q_rotary, k_rotary = rope.apply_qk(q, k, position_data)
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

    full_position_data = rope.prepare(q, position_ids)
    full_q, full_k = rope.apply_qk(q, k, full_position_data)
    cached_q, cached_k = zip(
        *[
            rope.apply_qk(
                q[:, :, index : index + 1],
                k[:, :, index : index + 1],
                rope.prepare(
                    q[:, :, index : index + 1],
                    position_ids[:, index : index + 1],
                ),
            )
            for index in range(position_ids.size(1))
        ]
    )

    torch.testing.assert_close(torch.cat(cached_q, dim=2), full_q)
    torch.testing.assert_close(torch.cat(cached_k, dim=2), full_k)


def test_prepare_and_apply_qk_compile_as_one_full_graph(rope: RotaryPE):
    q = torch.randn(2, NUM_HEAD, 4, HEAD_DIM)
    k = torch.randn_like(q)
    position_ids = torch.arange(4).repeat(2, 1)

    def apply_rope(query, key, positions):
        return rope.apply_qk(
            query,
            key,
            rope.prepare(query, positions),
        )

    compiled = torch.compile(apply_rope, backend="eager", fullgraph=True)
    expected = apply_rope(q, k, position_ids)
    actual = compiled(q, k, position_ids)

    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])


@pytest.mark.parametrize("use_scaled_dot_product_attention", [True, False])
def test_rope_preserves_half_precision_with_position_ids(
    use_scaled_dot_product_attention,
):
    rope = RotaryPE(HEAD_DIM // NUM_HEAD, SEQ_LEN, BASE)
    attention = CausalMultiHeadAttention(
        AttentionLayerConfig(
            context_length=SEQ_LEN,
            embedding_dimension=HEAD_DIM,
            num_heads=NUM_HEAD,
            attention_type=CausalMultiHeadAttention,
        ),
        qk_positional_embedding=rope,
    ).half()
    attention.has_flash_attn = use_scaled_dot_product_attention and hasattr(
        F,
        "scaled_dot_product_attention",
    )
    inputs = torch.randn(2, SEQ_LEN, HEAD_DIM, dtype=torch.float16)
    attention_mask = torch.ones(2, SEQ_LEN, dtype=torch.bool)
    position_ids = torch.arange(SEQ_LEN).repeat(2, 1)

    output = attention(
        inputs,
        attention_mask=attention_mask,
        qk_position_data=rope.prepare(inputs, position_ids),
    )

    assert output.dtype == torch.float16
    assert torch.isfinite(output).all()
