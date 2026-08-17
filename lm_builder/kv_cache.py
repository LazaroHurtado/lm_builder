import torch

from .utils import is_positive_integer


class KVCache:
    def __init__(self, capacity: int):
        if not is_positive_integer(capacity):
            raise ValueError("capacity must be a positive integer.")

        self.capacity = capacity
        self.k = None
        self.v = None
        self.key_mask = None
        self.active_length = 0

    def reset(self):
        self.active_length = 0
        if self.key_mask is not None:
            # Invalid entries are ignored, so the large K/V tensors need not be cleared.
            self.key_mask.zero_()

    def _initialize(self, k, v):
        if self.k is not None:
            return

        shape = (k.size(0), k.size(1), self.capacity, k.size(3))
        self.k = k.new_zeros(shape)
        self.v = v.new_zeros(shape)
        self.key_mask = torch.zeros(
            k.size(0),
            self.capacity,
            dtype=torch.bool,
            device=k.device,
        )

    def update(self, k, v, cache_position, attention_mask=None):
        if k.ndim != 4 or k.shape != v.shape:
            raise ValueError("Cached keys and values must have equal 4D shapes.")

        num_tokens = k.size(2)
        if not 0 < num_tokens <= self.capacity:
            raise ValueError("Cached sequence length must be between 1 and capacity.")
        if cache_position.ndim != 1 or cache_position.numel() != num_tokens:
            raise ValueError("Cache positions must match the cached sequence length.")

        self._initialize(k, v)
        cache_position = cache_position.to(device=k.device, dtype=torch.long)
        slots = cache_position.remainder(self.capacity)
        if attention_mask is None:
            attention_mask = torch.ones(
                k.size(0),
                num_tokens,
                dtype=torch.bool,
                device=k.device,
            )
        else:
            attention_mask = attention_mask[:, -num_tokens:].to(k.device).ne(0)

        self.k.index_copy_(2, slots, k)
        self.v.index_copy_(2, slots, v)
        self.key_mask.index_copy_(1, slots, attention_mask)

        self.active_length = min(
            self.active_length + num_tokens,
            self.capacity,
        )
        all_slots = torch.arange(self.active_length, device=k.device)
        latest_position = cache_position[-1]
        key_positions = latest_position - torch.remainder(
            latest_position - all_slots,
            self.capacity,
        )
        return (
            self.k[:, :, : self.active_length],
            self.v[:, :, : self.active_length],
            self.key_mask[:, : self.active_length],
            key_positions,
        )
