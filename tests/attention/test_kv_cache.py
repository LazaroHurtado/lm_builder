import pytest
import torch

from lm_builder.inference import KVCache


def test_kv_cache_appends_on_sequence_axis():
    cache = KVCache(context_length=4)
    first_key = torch.randn(2, 3, 2, 5)
    first_value = torch.randn(2, 3, 2, 5)
    second_key = torch.randn(2, 3, 1, 5)
    second_value = torch.randn(2, 3, 1, 5)

    cache.update(first_key, first_value)
    key, value = cache.update(second_key, second_value)

    assert cache.sequence_length == 3
    assert cache.tokens_seen == 3
    assert torch.equal(key, torch.cat((first_key, second_key), dim=2))
    assert torch.equal(value, torch.cat((first_value, second_value), dim=2))


def test_kv_cache_reset_reuses_allocated_storage():
    cache = KVCache(context_length=4)
    key = torch.randn(1, 2, 2, 3)
    value = torch.randn(1, 2, 2, 3)
    cache.update(key, value)
    key_storage = cache.k
    value_storage = cache.v

    cache.reset()
    assert cache.sequence_length == 0
    assert cache.tokens_seen == 0

    new_key = torch.randn(1, 2, 1, 3)
    new_value = torch.randn(1, 2, 1, 3)
    cached_key, cached_value = cache.update(new_key, new_value)

    assert cache.k is key_storage
    assert cache.v is value_storage
    assert cache.sequence_length == 1
    assert cache.tokens_seen == 1
    assert torch.equal(cached_key, new_key)
    assert torch.equal(cached_value, new_value)


def test_kv_cache_rolls_after_capacity_overflow():
    cache = KVCache(context_length=3)
    key = torch.arange(3, dtype=torch.float).view(1, 1, 3, 1)
    value = key + 10
    cache.update(key, value)
    key_storage = cache.k
    value_storage = cache.v

    cached_key, cached_value = cache.update(
        torch.tensor([[[[3.0]]]]),
        torch.tensor([[[[13.0]]]]),
    )

    assert cache.k is key_storage
    assert cache.v is value_storage
    assert cache.sequence_length == 3
    assert cache.tokens_seen == 4
    assert torch.equal(cached_key.flatten(), torch.tensor([1.0, 2.0, 3.0]))
    assert torch.equal(cached_value.flatten(), torch.tensor([11.0, 12.0, 13.0]))

    cached_key, cached_value = cache.update(
        torch.tensor([[[[4.0]]]]),
        torch.tensor([[[[14.0]]]]),
    )

    assert cache.tokens_seen == 5
    assert torch.equal(cached_key.flatten(), torch.tensor([2.0, 3.0, 4.0]))
    assert torch.equal(cached_value.flatten(), torch.tensor([12.0, 13.0, 14.0]))


def test_kv_cache_rejects_multi_token_rolling_update():
    cache = KVCache(context_length=2)
    cache.update(
        torch.randn(1, 1, 1, 3),
        torch.randn(1, 1, 1, 3),
    )

    with pytest.raises(ValueError, match="single-token updates"):
        cache.update(
            torch.randn(1, 1, 2, 3),
            torch.randn(1, 1, 2, 3),
        )


def test_kv_cache_rolls_with_context_length_one():
    cache = KVCache(context_length=1)
    cache.update(
        torch.tensor([[[[1.0]]]]),
        torch.tensor([[[[11.0]]]]),
    )

    key, value = cache.update(
        torch.tensor([[[[2.0]]]]),
        torch.tensor([[[[12.0]]]]),
    )

    assert cache.sequence_length == 1
    assert cache.tokens_seen == 2
    assert key.item() == 2
    assert value.item() == 12


@pytest.mark.parametrize("context_length", [None, 0, -1, 1.5, True])
def test_kv_cache_requires_positive_integer_context_length(context_length):
    with pytest.raises(ValueError, match="positive integer"):
        KVCache(context_length)
