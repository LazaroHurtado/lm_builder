from torch import nn

from .config import FeedForwardConfig


class FeedForward(nn.Module):
    """
    Source: https://arxiv.org/pdf/2002.05202v1.pdf
    """

    def __init__(self, config: FeedForwardConfig):
        super().__init__()

        self.in_dim = config.embedding_dimension
        self.hidden_dim = config.intermediate_dimension

        self.up_proj = nn.Linear(self.in_dim, self.hidden_dim, bias=config.bias)
        self.activation_fn = config.activation_fn()
        self.down_proj = nn.Linear(self.hidden_dim, self.in_dim, bias=config.bias)
        self.gate_proj = nn.Linear(self.in_dim, self.hidden_dim, bias=config.bias)

        self.config = config

    def forward(self, x):
        out1 = self.up_proj(x)
        out2 = self.gate_proj(x)

        x = out1 * self.activation_fn(out2)

        out3 = self.down_proj(x)
        return out3
