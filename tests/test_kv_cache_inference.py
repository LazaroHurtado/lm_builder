import pytest
import torch
from torch.nn import functional as F

from lm_builder import LanguageModel
from lm_builder.attention import (
    AttentionConfig,
    CausalMultiHeadAttention,
    GroupedQueryAttention,
    MultiHeadAttention,
    MultiQueryAttention,
)
from lm_builder.ffn import FeedForward, FeedForwardConfig
from lm_builder.inference import KVCache
from lm_builder.positional_embeddings import AbsolutePE, RotaryPE
from lm_builder.transformer import TransformerConfig


def build_model(
    attention_type=CausalMultiHeadAttention,
    position_type="rotary",
    context_length=6,
    num_layers=2,
    ratio=None,
    window_size=3,
    qk_norm=None,
):
    attention_types = (
        attention_type if isinstance(attention_type, list) else [attention_type]
    )
    window_sizes = (
        window_size
        if isinstance(window_size, list)
        else [window_size] * len(attention_types)
    )
    attention_config = AttentionConfig.build_configs(
        {
            "num_heads": 4,
            "positional_embedding": (RotaryPE if position_type == "rotary" else None),
            "qk_norm": qk_norm,
            "layers": [
                {
                    "type": resolved_attention_type,
                    "kv_heads": 2,
                    "window_size": resolved_window_size,
                }
                for resolved_attention_type, resolved_window_size in zip(
                    attention_types,
                    window_sizes,
                )
            ],
            "ratio": ratio,
        },
        num_layers,
        context_length,
        8,
    )
    config = TransformerConfig(
        embedding_dimension=8,
        context_length=context_length,
        attention_config=attention_config,
        ffn_config=FeedForwardConfig(
            embedding_dimension=8,
            intermediate_dimension=16,
            ffn_type=FeedForward,
        ),
        vocab_size=16,
        num_layers=num_layers,
        positional_embedding=AbsolutePE if position_type == "absolute" else None,
    )
    return LanguageModel(config, tokenizer=None).eval()


@pytest.mark.parametrize(
    "attention_type",
    [
        CausalMultiHeadAttention,
        MultiQueryAttention,
        GroupedQueryAttention,
    ],
)
@pytest.mark.parametrize("position_type", ["absolute", "rotary"])
@pytest.mark.parametrize("use_scaled_dot_product_attention", [True, False])
@pytest.mark.parametrize("with_attention_mask", [True, False])
@pytest.mark.parametrize("with_qk_norm", [True, False])
def test_cached_decode_matches_full_forward(
    attention_type,
    position_type,
    use_scaled_dot_product_attention,
    with_attention_mask,
    with_qk_norm,
):
    torch.manual_seed(7)
    model = build_model(
        attention_type=attention_type,
        position_type=position_type,
        qk_norm={"type": "RMSNorm"} if with_qk_norm else None,
    )
    for block in model.transformer.blocks:
        block.attn.has_flash_attn = use_scaled_dot_product_attention and hasattr(
            F, "scaled_dot_product_attention"
        )

    input_ids = torch.tensor([[0, 2, 3, 4], [5, 6, 7, 8]])
    attention_mask = (
        torch.tensor([[0, 1, 1, 1], [1, 1, 1, 1]]) if with_attention_mask else None
    )
    kv_caches = [KVCache(model.context_length) for _ in model.transformer.blocks]

    with torch.inference_mode():
        full_logits, _, _ = model(input_ids, attention_mask=attention_mask)
        model(
            input_ids[:, :-1],
            attention_mask=(
                attention_mask[:, :-1] if attention_mask is not None else None
            ),
            _kv_caches=kv_caches,
        )
        cached_logits, _, _ = model(
            input_ids[:, -1:],
            attention_mask=attention_mask,
            _kv_caches=kv_caches,
        )

    assert torch.allclose(
        cached_logits[:, -1],
        full_logits[:, -1],
        atol=1e-5,
        rtol=1e-5,
    )


def test_generate_prefills_once_then_decodes_single_tokens():
    torch.manual_seed(11)
    model = build_model(context_length=8)
    projected_sequence_lengths = []
    hook = model.transformer.blocks[0].attn.qkv_proj.register_forward_pre_hook(
        lambda _, inputs: projected_sequence_lengths.append(inputs[0].size(1))
    )

    try:
        list(
            model.generate(
                torch.tensor([[1, 2, 3]]),
                max_new_tokens=3,
                temperature=0,
            )
        )
        assert projected_sequence_lengths == [3, 1, 1]

        projected_sequence_lengths.clear()
        list(
            model.generate(
                torch.tensor([[1, 2, 3]]),
                max_new_tokens=3,
                temperature=0,
                use_cache=False,
            )
        )
        assert projected_sequence_lengths == [3, 4, 5]
    finally:
        hook.remove()


@pytest.mark.parametrize(
    "attention_type",
    [
        CausalMultiHeadAttention,
        MultiQueryAttention,
        GroupedQueryAttention,
    ],
)
def test_absolute_position_cache_recomputes_across_context_overflow(attention_type):
    torch.manual_seed(13)
    model = build_model(
        attention_type=attention_type,
        position_type="absolute",
        context_length=4,
    )
    input_ids = torch.tensor([[0, 2, 3], [4, 5, 6]])
    attention_mask = torch.tensor([[0, 1, 1], [1, 1, 1]])
    projected_sequence_lengths = []
    hook = model.transformer.blocks[0].attn.qkv_proj.register_forward_pre_hook(
        lambda _, inputs: projected_sequence_lengths.append(inputs[0].size(1))
    )

    try:
        cached_tokens = torch.cat(
            list(
                model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=5,
                    temperature=0,
                )
            ),
            dim=1,
        )
    finally:
        hook.remove()

    uncached_tokens = torch.cat(
        list(
            model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=5,
                temperature=0,
                use_cache=False,
            )
        ),
        dim=1,
    )

    assert torch.equal(cached_tokens, uncached_tokens)
    assert projected_sequence_lengths == [3, 1, 4, 4, 4]


@pytest.mark.parametrize(
    "attention_type",
    [
        CausalMultiHeadAttention,
        MultiQueryAttention,
        GroupedQueryAttention,
    ],
)
@pytest.mark.parametrize("position_type", ["rotary", None])
def test_rolling_cache_keeps_single_token_decoding_after_overflow(
    attention_type,
    position_type,
):
    model = build_model(
        attention_type=attention_type,
        position_type=position_type,
        context_length=4,
    )
    projected_sequence_lengths = []
    hook = model.transformer.blocks[0].attn.qkv_proj.register_forward_pre_hook(
        lambda _, inputs: projected_sequence_lengths.append(inputs[0].size(1))
    )

    try:
        list(
            model.generate(
                torch.tensor([[1, 2, 3]]),
                attention_mask=torch.ones(1, 3, dtype=torch.long),
                max_new_tokens=5,
                temperature=0,
            )
        )
    finally:
        hook.remove()

    assert projected_sequence_lengths == [3, 1, 1, 1, 1]


@pytest.mark.parametrize(
    "attention_type",
    [
        CausalMultiHeadAttention,
        MultiQueryAttention,
        GroupedQueryAttention,
    ],
)
@pytest.mark.parametrize("with_attention_mask", [True, False])
def test_rolling_cache_handles_prompt_longer_than_context(
    attention_type,
    with_attention_mask,
):
    torch.manual_seed(19)
    model = build_model(
        attention_type=attention_type,
        position_type="rotary",
        context_length=4,
        num_layers=1,
    )
    input_ids = torch.tensor([[0, 0, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]])
    attention_mask = (
        torch.tensor([[0, 0, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]])
        if with_attention_mask
        else None
    )

    cached_tokens = torch.cat(
        list(
            model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=3,
                temperature=0,
            )
        ),
        dim=1,
    )
    uncached_tokens = torch.cat(
        list(
            model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=3,
                temperature=0,
                use_cache=False,
            )
        ),
        dim=1,
    )

    assert torch.equal(cached_tokens, uncached_tokens)


def test_rolling_cache_handles_context_length_one():
    model = build_model(
        position_type="rotary",
        context_length=1,
    )
    projected_sequence_lengths = []
    hook = model.transformer.blocks[0].attn.qkv_proj.register_forward_pre_hook(
        lambda _, inputs: projected_sequence_lengths.append(inputs[0].size(1))
    )

    try:
        cached_tokens = torch.cat(
            list(
                model.generate(
                    torch.tensor([[1]]),
                    max_new_tokens=3,
                    temperature=0,
                )
            ),
            dim=1,
        )
    finally:
        hook.remove()

    uncached_tokens = torch.cat(
        list(
            model.generate(
                torch.tensor([[1]]),
                max_new_tokens=3,
                temperature=0,
                use_cache=False,
            )
        ),
        dim=1,
    )

    assert torch.equal(cached_tokens, uncached_tokens)
    assert projected_sequence_lengths == [1, 1, 1]


def test_rolling_cache_supports_mixed_attention_layers():
    model = build_model(
        attention_type=[GroupedQueryAttention, CausalMultiHeadAttention],
        position_type="rotary",
        context_length=4,
        num_layers=2,
        ratio=[1, 1],
        window_size=[2, None],
    )
    projected_sequence_lengths = [[], []]
    hooks = [
        block.attn.qkv_proj.register_forward_pre_hook(
            lambda _, inputs, layer_index=layer_index: projected_sequence_lengths[
                layer_index
            ].append(inputs[0].size(1))
        )
        for layer_index, block in enumerate(model.transformer.blocks)
    ]

    try:
        list(
            model.generate(
                torch.tensor([[1, 2, 3]]),
                max_new_tokens=5,
                temperature=0,
            )
        )
    finally:
        for hook in hooks:
            hook.remove()

    assert projected_sequence_lengths == [
        [3, 1, 1, 1, 1],
        [3, 1, 1, 1, 1],
    ]


def test_generation_uses_fresh_cache_for_each_call():
    torch.manual_seed(17)
    model = build_model(context_length=8)
    projected_sequence_lengths = []
    hook = model.transformer.blocks[0].attn.qkv_proj.register_forward_pre_hook(
        lambda _, inputs: projected_sequence_lengths.append(inputs[0].size(1))
    )

    try:
        for _ in range(2):
            list(
                model.generate(
                    torch.tensor([[1, 2, 3]]),
                    max_new_tokens=2,
                    temperature=0,
                )
            )
    finally:
        hook.remove()

    assert projected_sequence_lengths == [3, 1, 3, 1]


def test_grouped_query_cache_keeps_native_kv_heads():
    model = build_model(attention_type=GroupedQueryAttention)
    kv_caches = [KVCache(model.context_length) for _ in model.transformer.blocks]

    with torch.inference_mode():
        model(
            torch.tensor([[1, 2, 3]]),
            _kv_caches=kv_caches,
        )

    assert all(cache.k.size(1) == 2 for cache in kv_caches)


def test_generate_requires_eval_mode_for_kv_cache():
    model = build_model().train()

    with pytest.raises(AssertionError, match="eval mode"):
        list(
            model.generate(
                torch.tensor([[1, 2, 3]]),
                max_new_tokens=1,
                temperature=0,
            )
        )


def test_generate_rejects_noncausal_attention_cache():
    model = build_model(
        attention_type=MultiHeadAttention,
        window_size=None,
    )

    with pytest.raises(AssertionError, match="requires causal attention"):
        list(
            model.generate(
                torch.tensor([[1, 2, 3]]),
                max_new_tokens=1,
                temperature=0,
            )
        )

    tokens = list(
        model.generate(
            torch.tensor([[1, 2, 3]]),
            max_new_tokens=1,
            temperature=0,
            use_cache=False,
        )
    )
    assert len(tokens) == 1
