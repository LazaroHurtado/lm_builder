import torch


class RotaryPE:
    # For rotary positional embedding, we take chunks of two from the
    # token embeddings and apply a rotation.

    def __init__(
        self,
        embedding_dim: int,
        context_len: int,
        base: float = 10_000.0,
        **_kwargs,
    ):
        if embedding_dim % 2 != 0:
            embedding_dim += 1

        self.embedding_dim = embedding_dim
        self.context_len = context_len
        self.base = base

        power = (
            torch.arange(
                0,
                self.embedding_dim,
                step=2,
                device="cpu",
                dtype=torch.float32,
            )
            / self.embedding_dim
        )
        self.inv_freq = 1.0 / (self.base**power)

    @torch.no_grad()
    def prepare(self, x: torch.Tensor, position_ids: torch.Tensor):
        assert (
            x.device.type != "meta" and position_ids.device.type != "meta"
        ), "RotaryPE does not support meta tensors."

        batch_size = position_ids.shape[0]
        inv_freq = self.inv_freq.to(device=x.device, dtype=torch.float32)
        inv_freq_expanded = inv_freq[None, :, None].expand(batch_size, -1, 1)
        position_ids_expanded = position_ids[:, None, :].to(
            device=x.device,
            dtype=torch.float32,
        )

        device_type = x.device.type if x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    @staticmethod
    def rotate_half(x: torch.Tensor):
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

    @staticmethod
    def apply_qk(
        q: torch.Tensor,
        k: torch.Tensor,
        position_data,
        unsqueeze_dim=1,
    ):
        cos, sin = position_data
        cos = cos.unsqueeze(unsqueeze_dim).to(dtype=q.dtype)
        sin = sin.unsqueeze(unsqueeze_dim).to(dtype=q.dtype)
        q_embed = (q * cos) + (RotaryPE.rotate_half(q) * sin)
        k_embed = (k * cos) + (RotaryPE.rotate_half(k) * sin)
        return q_embed, k_embed
