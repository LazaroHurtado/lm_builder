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
    new_key = torch.randn(1, 2, 1, 3)
    new_value = torch.randn(1, 2, 1, 3)
    cached_key, cached_value = cache.update(new_key, new_value)

    assert cache.k is key_storage
    assert cache.v is value_storage
    assert cache.sequence_length == 1
    assert torch.equal(cached_key, new_key)
    assert torch.equal(cached_value, new_value)


def test_kv_cache_rejects_capacity_overflow():
    cache = KVCache(context_length=2)
    key = torch.randn(1, 1, 2, 3)
    value = torch.randn(1, 1, 2, 3)
    cache.update(key, value)

    with pytest.raises(ValueError, match="capacity exceeded"):
        cache.update(key[:, :, :1], value[:, :, :1])


@pytest.mark.parametrize("context_length", [None, 0, -1, 1.5, True])
def test_kv_cache_requires_positive_integer_context_length(context_length):
    with pytest.raises(ValueError, match="positive integer"):
        KVCache(context_length)
