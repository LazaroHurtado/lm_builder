from torch import nn

from ..utils import select_positional_embedding


class LearnablePE(nn.Embedding):
    def __init__(self, context_length: int, embedding_dim: int, _base: float):
        super().__init__(context_length, embedding_dim)

    def forward(self, input, position_ids=None):  # pylint: disable=redefined-builtin
        return input + select_positional_embedding(self.weight, input, position_ids)
