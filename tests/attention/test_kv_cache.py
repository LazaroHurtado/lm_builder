import pytest
import torch

from lm_builder.kv_cache import KVCache


def test_kv_cache_reuses_fixed_storage_and_returns_populated_views():
    cache = KVCache(capacity=4)
    first_key = torch.randn(2, 3, 2, 5)
    first_value = torch.randn(2, 3, 2, 5)
    second_key = torch.randn(2, 3, 1, 5)
    second_value = torch.randn(2, 3, 1, 5)

    first_cached_key, first_cached_value, first_mask, _ = cache.update(
        first_key,
        first_value,
        torch.tensor([0, 1]),
    )
    key_storage = cache.k
    value_storage = cache.v
    key, value, key_mask, _ = cache.update(
        second_key,
        second_value,
        torch.tensor([2]),
    )

    assert cache.k is key_storage
    assert cache.v is value_storage
    assert cache.k.size(2) == cache.capacity == 4
    assert cache.v.size(2) == 4
    assert first_cached_key.size(2) == 2
    assert first_cached_value.size(2) == 2
    assert first_mask.size(1) == 2
    assert key.size(2) == 3
    assert value.size(2) == 3
    assert key_mask.size(1) == 3
    assert key.untyped_storage().data_ptr() == cache.k.untyped_storage().data_ptr()
    assert value.untyped_storage().data_ptr() == cache.v.untyped_storage().data_ptr()
    assert torch.equal(key, torch.cat((first_key, second_key), dim=2))
    assert torch.equal(value, torch.cat((first_value, second_value), dim=2))
    assert key_mask.all()


def test_kv_cache_reset_reuses_allocated_storage():
    cache = KVCache(capacity=4)
    key = torch.randn(1, 2, 2, 3)
    value = torch.randn(1, 2, 2, 3)
    cache.update(key, value, torch.tensor([0, 1]))
    key_storage = cache.k
    value_storage = cache.v

    cache.reset()
    assert not cache.key_mask.any()

    new_key = torch.randn(1, 2, 1, 3)
    new_value = torch.randn(1, 2, 1, 3)
    cached_key, cached_value, _, _ = cache.update(
        new_key,
        new_value,
        torch.tensor([0]),
    )

    assert cache.k is key_storage
    assert cache.v is value_storage
    assert cached_key.size(2) == 1
    assert cached_value.size(2) == 1
    assert torch.equal(cached_key[:, :, :1], new_key)
    assert torch.equal(cached_value[:, :, :1], new_value)


def test_kv_cache_rolls_after_capacity_overflow():
    cache = KVCache(capacity=3)
    key = torch.arange(3, dtype=torch.float).view(1, 1, 3, 1)
    value = key + 10
    cache.update(key, value, torch.tensor([0, 1, 2]))
    key_storage = cache.k
    value_storage = cache.v

    cached_key, cached_value, _, key_positions = cache.update(
        torch.tensor([[[[3.0]]]]),
        torch.tensor([[[[13.0]]]]),
        torch.tensor([3]),
    )

    assert cache.k is key_storage
    assert cache.v is value_storage
    assert torch.equal(cached_key.flatten(), torch.tensor([3.0, 1.0, 2.0]))
    assert torch.equal(cached_value.flatten(), torch.tensor([13.0, 11.0, 12.0]))
    assert torch.equal(key_positions, torch.tensor([3, 1, 2]))

    cached_key, cached_value, _, key_positions = cache.update(
        torch.tensor([[[[4.0]]]]),
        torch.tensor([[[[14.0]]]]),
        torch.tensor([4]),
    )

    assert torch.equal(cached_key.flatten(), torch.tensor([3.0, 4.0, 2.0]))
    assert torch.equal(cached_value.flatten(), torch.tensor([13.0, 14.0, 12.0]))
    assert torch.equal(key_positions, torch.tensor([3, 4, 2]))


def test_kv_cache_tracks_padding():
    cache = KVCache(capacity=4)
    key = torch.randn(2, 1, 3, 2)
    value = torch.randn(2, 1, 3, 2)
    attention_mask = torch.tensor([[0, 1, 1], [1, 1, 1]])

    _, _, key_mask, _ = cache.update(
        key,
        value,
        torch.tensor([0, 1, 2]),
        attention_mask,
    )

    assert torch.equal(
        key_mask,
        torch.tensor(
            [
                [False, True, True],
                [True, True, True],
            ]
        ),
    )


def test_kv_cache_rolls_with_multi_token_update():
    cache = KVCache(capacity=4)
    cache.update(
        torch.tensor([[[[0.0], [1.0], [2.0]]]]),
        torch.tensor([[[[10.0], [11.0], [12.0]]]]),
        torch.tensor([0, 1, 2]),
    )

    key, value, _, _ = cache.update(
        torch.tensor([[[[3.0], [4.0]]]]),
        torch.tensor([[[[13.0], [14.0]]]]),
        torch.tensor([3, 4]),
    )

    assert torch.equal(key.flatten(), torch.tensor([4.0, 1.0, 2.0, 3.0]))
    assert torch.equal(value.flatten(), torch.tensor([14.0, 11.0, 12.0, 13.0]))


def test_kv_cache_rolls_with_capacity_one():
    cache = KVCache(capacity=1)
    cache.update(
        torch.tensor([[[[1.0]]]]),
        torch.tensor([[[[11.0]]]]),
        torch.tensor([0]),
    )

    key, value, _, _ = cache.update(
        torch.tensor([[[[2.0]]]]),
        torch.tensor([[[[12.0]]]]),
        torch.tensor([1]),
    )

    assert key.item() == 2
    assert value.item() == 12


@pytest.mark.parametrize("capacity", [None, 0, -1, 1.5, True])
def test_kv_cache_requires_positive_integer_capacity(capacity):
    with pytest.raises(ValueError, match="positive integer"):
        KVCache(capacity)
