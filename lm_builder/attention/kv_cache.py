import torch


class KVCache:
    def __init__(self, context_length: int):
        self.k = None
        self.v = None

        self.first_prefill = True
        self.context_length = context_length

    def update(self, k, v):
        if self.k is None or self.v is None:
            self.k = k
            self.v = v
        else:
            self.k = torch.cat((self.k, k), dim=1)[:, -self.context_length :, :, :]
            self.v = torch.cat((self.v, v), dim=1)[:, -self.context_length :, :, :]

        return self.k, self.v
