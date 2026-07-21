class KVCache:
    def __init__(self, context_length: int):
        if (
            not isinstance(context_length, int)
            or isinstance(context_length, bool)
            or context_length <= 0
        ):
            raise ValueError("context_length must be a positive integer.")

        self.k = None
        self.v = None
        self.context_length = context_length
        self._sequence_length = 0

    @property
    def sequence_length(self):
        return self._sequence_length

    def __len__(self):
        return self.sequence_length

    def reset(self):
        self._sequence_length = 0

    def _storage_matches(self, storage, tensor):
        return (
            storage is not None
            and storage.shape
            == (
                tensor.size(0),
                tensor.size(1),
                self.context_length,
                tensor.size(3),
            )
            and storage.device == tensor.device
            and storage.dtype == tensor.dtype
        )

    def _ensure_storage(self, k, v):
        if self._storage_matches(self.k, k) and self._storage_matches(self.v, v):
            return

        if self.sequence_length:
            raise ValueError(
                "Cached key/value shape, device, or dtype changed before reset."
            )

        self.k = k.new_empty(
            k.size(0),
            k.size(1),
            self.context_length,
            k.size(3),
        )
        self.v = v.new_empty(
            v.size(0),
            v.size(1),
            self.context_length,
            v.size(3),
        )

    def update(self, k, v):
        if k.ndim != 4 or v.ndim != 4:
            raise ValueError("Cached keys and values must be four-dimensional.")
        if k.shape[:3] != v.shape[:3]:
            raise ValueError(
                "Cached keys and values must share batch, head, and sequence shapes."
            )
        if k.device != v.device or k.dtype != v.dtype:
            raise ValueError("Cached keys and values must share device and dtype.")

        next_sequence_length = self.sequence_length + k.size(2)
        if next_sequence_length > self.context_length:
            raise ValueError("KV cache capacity exceeded; reset before updating.")

        self._ensure_storage(k, v)
        start = self.sequence_length
        self.k[:, :, start:next_sequence_length].copy_(k)
        self.v[:, :, start:next_sequence_length].copy_(v)
        self._sequence_length = next_sequence_length

        return (
            self.k[:, :, :next_sequence_length],
            self.v[:, :, :next_sequence_length],
        )
