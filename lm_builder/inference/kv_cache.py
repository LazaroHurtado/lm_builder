from ..utils import is_positive_integer


class KVCache:
    def __init__(self, context_length: int):
        if not is_positive_integer(context_length):
            raise ValueError("context_length must be a positive integer.")

        self.k = None
        self.v = None
        self.context_length = context_length
        self._sequence_length = 0
        self._tokens_seen = 0

    @property
    def sequence_length(self):
        return self._sequence_length

    @property
    def tokens_seen(self):
        return self._tokens_seen

    def __len__(self):
        return self.sequence_length

    def reset(self):
        self._sequence_length = 0
        self._tokens_seen = 0

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

    def _shift_left(self, positions):
        retained_length = self.sequence_length - positions
        if retained_length:
            self.k[:, :, :retained_length].copy_(
                self.k[:, :, positions : self.sequence_length].clone()
            )
            self.v[:, :, :retained_length].copy_(
                self.v[:, :, positions : self.sequence_length].clone()
            )
        return retained_length

    def update(self, k, v):
        if k.ndim != 4 or v.ndim != 4:
            raise ValueError("Cached keys and values must be four-dimensional.")
        if k.shape[:3] != v.shape[:3]:
            raise ValueError(
                "Cached keys and values must share batch, head, and sequence shapes."
            )
        if k.device != v.device or k.dtype != v.dtype:
            raise ValueError("Cached keys and values must share device and dtype.")

        incoming_length = k.size(2)
        next_sequence_length = self.sequence_length + incoming_length
        if next_sequence_length > self.context_length and incoming_length != 1:
            raise ValueError(
                "Rolling KV cache overflow only supports single-token updates."
            )

        self._ensure_storage(k, v)
        overflow = max(0, next_sequence_length - self.context_length)
        start = self._shift_left(overflow) if overflow else self.sequence_length
        end = start + incoming_length
        self.k[:, :, start:end].copy_(k)
        self.v[:, :, start:end].copy_(v)
        self._sequence_length = min(next_sequence_length, self.context_length)
        self._tokens_seen += incoming_length

        return (
            self.k[:, :, : self.sequence_length],
            self.v[:, :, : self.sequence_length],
        )
