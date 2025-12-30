import torch
from torch import nn


class RotaryPE(nn.Module):
    # For rotary positional embedding, we take chunks of two from the
    # token embeddings and apply a rotation.

    def __init__(self, embedding_dim: int, context_len: int, base: float, **kwargs):
        super().__init__()

        if embedding_dim % 2 != 0:
            embedding_dim += 1

        self.embedding_dim = embedding_dim
        self.context_len = context_len
        self.base = base

        self.cos, self.sin = None, None

        self._generate_positional_embeddings()

    def _generate_positional_embeddings(self):
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
            torch.arange(0, self.embedding_dim, step=2, dtype=torch.float)
            / self.embedding_dim
        )
        inv_freq = 1.0 / (self.base**power)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(self.context_len, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def _apply(self, fn):
        if self.inv_freq.device.type == "meta":
            self._generate_positional_embeddings()
        return super()._apply(fn)

    def _get_cos_sin_embeddings(self, position_ids, device, unsqueeze_dim=1):
        C = position_ids.shape[0]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(C, -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()

        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)

        self.cos = emb.cos().unsqueeze(unsqueeze_dim).to(device)
        self.sin = emb.sin().unsqueeze(unsqueeze_dim).to(device)

        return self.cos, self.sin

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

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        unsqueeze_dim=1,
    ):
        # q: (B, H, T, D)
        T = q.shape[2]

        # Index into the cached tables
        # self.cos_cached: (MaxT, D) -> (B, T, D) via slicing
        cos = self.cos_cached[:T].unsqueeze(0).unsqueeze(unsqueeze_dim)
        sin = self.sin_cached[:T].unsqueeze(0).unsqueeze(unsqueeze_dim)

        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        return q_embed, k_embed
