from torch import nn


class LearnablePE(nn.Embedding):
    def __init__(self, context_length: int, embedding_dim: int, _base: float):
        super().__init__(context_length, embedding_dim)

    def forward(self, input, position_ids=None):  # pylint: disable=redefined-builtin
        _, T, C = input.size()  # pylint: disable=invalid-name
        if position_ids is None:
            positional_embedding = self.weight[None, :T, :C]
        else:
            positional_embedding = super().forward(position_ids)[..., :C]

        return input + positional_embedding
