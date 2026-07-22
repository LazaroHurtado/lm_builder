import torch


class RotaryPE:
    # For rotary positional embedding, we take chunks of two from the
    # token embeddings and apply a rotation.

    KEY_TO_INSTANCE = {}

    def __new__(cls, embedding_dim: int, context_len: int, base: float, **_kwargs):
        key = (embedding_dim, context_len, base)
        if key not in cls.KEY_TO_INSTANCE:
            cls.KEY_TO_INSTANCE[key] = super().__new__(cls)
        return cls.KEY_TO_INSTANCE[key]

    def __init__(self, embedding_dim: int, context_len: int, base: float, **_kwargs):
        if getattr(self, "_initialized", False):
            return

        if embedding_dim % 2 != 0:
            embedding_dim += 1

        self.embedding_dim = embedding_dim
        self.context_len = context_len
        self.base = base

        self.inv_freq = None
        self.cos_cached = None
        self.sin_cached = None

        self._initialized = True

    def _generate_positional_embeddings(self, device, dtype):
        # In the ReFormer paper, the positional embedding is applied to
        # Q and K matrices in the attention layer. It does so by taking
        # chunks of two from the token embeddings and applying a rotation
        # matrix

        # Since we are taking chunks of two from the token embeddings, we
        # start with half the size of the embedding dimension and then repeat
        # it twice to match the embedding dimension. In the ReFormer paper,
        # this is called the thetas, but we will refer to it as the inverse
        # frequency, inv_freq
        # (C)
        power = (
            torch.arange(
                0,
                self.embedding_dim,
                step=2,
                device=device,
                dtype=torch.float32,
            )
            / self.embedding_dim
        )
        inv_freq = 1.0 / (self.base**power)

        t = torch.arange(self.context_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        self.inv_freq = inv_freq
        self.cos_cached = emb.cos().to(dtype=dtype)
        self.sin_cached = emb.sin().to(dtype=dtype)

    def _get_cos_sin_embeddings(self, position_ids, device, unsqueeze_dim=1):
        batch_size = position_ids.shape[0]
        inv_freq = self.inv_freq.to(device=device, dtype=torch.float32)
        inv_freq_expanded = inv_freq[None, :, None].expand(batch_size, -1, 1)
        position_ids_expanded = position_ids[:, None, :].to(
            device=device,
            dtype=torch.float32,
        )

        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)

        return emb.cos().unsqueeze(unsqueeze_dim), emb.sin().unsqueeze(unsqueeze_dim)

    def rotate_half(self, x: torch.Tensor):
        # Helper function to rotate the last dimension of x by half. This differs
        # from the math notation in the RoPE paper, which uses pairs [(x_1, x_2), (x_3, x_4)],
        # but it is equivalent.
        # Example:
        #   Assume we have a tensor x with shape (B, T, C) where C=4
        #   [[1.0, 2.0, 3.0, 4.0]]
        #   We want to split this tensor into two halves:
        #   first half:  [1.0, 2.0]
        #   second half: [3.0, 4.0]
        #   Then we want to rotate these halves such that:
        #   rotated: [-2.0, -3.0, 0.0, 1.0]
        #   This is equivalent to taking the negative of the odd indices
        #   and swapping them with the even indices.
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        unsqueeze_dim=1,
        position_ids=None,
    ):
        assert (
            q.device.type != "meta" and k.device.type != "meta"
        ), "RotaryPE does not support meta tensors."
        if position_ids is not None:
            assert (
                position_ids.device.type != "meta"
            ), "RotaryPE does not support meta tensors."

        # q: (B, H, T, D)
        T = q.shape[2]

        if self.inv_freq is None:
            self._generate_positional_embeddings(device=q.device, dtype=q.dtype)

        if position_ids is None:
            cos = self.cos_cached[:T].unsqueeze(0).unsqueeze(unsqueeze_dim)
            sin = self.sin_cached[:T].unsqueeze(0).unsqueeze(unsqueeze_dim)
        elif bool(((position_ids >= 0) & (position_ids < self.context_len)).all()):
            cached_position_ids = position_ids.to(
                device=q.device,
                dtype=torch.long,
            )
            cos = self.cos_cached.to(device=q.device, dtype=q.dtype)[
                cached_position_ids
            ].unsqueeze(unsqueeze_dim)
            sin = self.sin_cached.to(device=q.device, dtype=q.dtype)[
                cached_position_ids
            ].unsqueeze(unsqueeze_dim)
        else:
            cos, sin = self._get_cos_sin_embeddings(
                position_ids,
                q.device,
                unsqueeze_dim=unsqueeze_dim,
            )

        cos = cos.to(device=q.device, dtype=q.dtype)
        sin = sin.to(device=q.device, dtype=q.dtype)
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        return q_embed, k_embed
