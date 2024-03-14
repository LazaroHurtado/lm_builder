from .config import FeedForwardConfig

from torch import nn


class FeedForward(nn.Module):
    """
    Q1: Why do we set bias to false?
    Q2: Why are there three linear layers?
    Q3: Why are we doing (act(x@W2) * (x@W1))@W3?

    Answers to all theses questions can be found in this paper:
    https://arxiv.org/pdf/2002.05202v1.pdf
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
