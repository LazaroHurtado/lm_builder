from .config import FeedForwardConfig

from torch import nn


class ClassicFeedForward(nn.Module):
    # The typical ffn with a single hidden layer and two weight matrices

    def __init__(self, config: FeedForwardConfig):
        super().__init__()

        self.in_dim = config.embedding_dimension
        self.hidden_dim = config.intermediate_dimension

        self.up_proj = nn.Linear(self.in_dim, self.hidden_dim)
        self.activation_fn = config.activation_fn()
        self.down_proj = nn.Linear(self.hidden_dim, self.in_dim)
        self.dropout = nn.Dropout(config.dropout)

        self.config = config

    def forward(self, x):
        x = self.up_proj(x)
        x = self.activation_fn(x)

        x = self.down_proj(x)
        x = self.dropout(x)

        return x
