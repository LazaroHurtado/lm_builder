import torch


class AbsolutePE:
    # For absolute positional embedding, half of the embeddings comes from a
    # sin wave and the other half comes from a cos wave.

    KEY_TO_INSTANCE = {}

    def __new__(cls, context_length: int, embedding_dim: int, base: float):
        key = (context_length, embedding_dim, base)
        if key not in cls.KEY_TO_INSTANCE:
            cls.KEY_TO_INSTANCE[key] = super().__new__(cls)
        return cls.KEY_TO_INSTANCE[key]

    def __init__(self, context_length: int, embedding_dim: int, base: float):
        if getattr(self, "_initialized", False):
            return

        if embedding_dim % 2 != 0:
            embedding_dim += 1

        self.embedding_dim = embedding_dim
        self.context_length = context_length
        self.base = base
        self.weight = None
        self._initialized = True

    def _generate_positional_embeddings(self, device, dtype):
        # We step by 2 so we can generate the sin and cos, which each takes half the
        # total embedding dimension, waves separately and then stack them together.
        power = (
            2
            * torch.arange(
                0,
                self.embedding_dim,
                step=2,
                device=device,
                dtype=torch.float32,
            )
            / self.embedding_dim
        )
        inv_freq = 1 / (self.base**power)
        pos = torch.arange(
            self.context_length,
            device=device,
            dtype=torch.float32,
        )
        angles = torch.outer(pos, inv_freq)
        sinusoids = torch.stack((angles.sin().view(-1), angles.cos().view(-1)))
        self.weight = self.interleave(
            sinusoids,
            (self.context_length, self.embedding_dim),
        ).to(dtype=dtype)

    def interleave(self, x: torch.Tensor, shape: torch.Size):
        # I will explain this through an example:
        # Example:
        #   Assume T=2 and C=3 and x is the following tensor
        #   [[0.0, 0.2, 0.4],   <- sin
        #    [1.0, 0.5, 0.1]]   <- cos
        #   then what we want is the following
        #   [[0.0, 1.0, 0.2],
        #    [0.5, 0.4, 0.1]]
        #   which interleaves the sin and cos values. We can get
        #   this behavior by first transposing the x tensor
        #   [[0.0, 1.0]
        #    [0.2, 0.5]
        #    [0.4, 0.1]]
        #   getting a contiguous view
        #   [0.0, 1.0, 0.2, 0.5, 0.4, 0.1]
        #   and then reshaping it to the desired shape we get the desired result
        #   [[0.0, 1.0, 0.2],
        #    [0.5, 0.4, 0.1]]
        # (2, T*C/2) -> (T*C/2, 2) -> (T*C) -> (T, C)
        return x.t().contiguous().view(*shape)

    def __call__(self, x: torch.Tensor, position_ids=None):
        assert x.device.type != "meta", "AbsolutePE does not support meta tensors."
        if position_ids is not None:
            assert (
                position_ids.device.type != "meta"
            ), "AbsolutePE does not support meta tensors."

        _, T, C = x.size()
        if self.weight is None:
            self._generate_positional_embeddings(device=x.device, dtype=x.dtype)

        weight = self.weight.to(device=x.device, dtype=x.dtype)
        if position_ids is None:
            positional_embedding = weight[None, :T, :C]
        else:
            position_ids = position_ids.to(device=x.device, dtype=torch.long)
            positional_embedding = weight[position_ids, :C]

        return x + positional_embedding
