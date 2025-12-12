import torch
from torch import nn

from .config import FeedForwardConfig
from .feed_forward import FeedForward


class MixtureOfExperts(nn.Module):

    def __init__(self, config: FeedForwardConfig):
        super().__init__()

        self.embedding_dim = config.embedding_dimension
        self.intermediate_dim = config.intermediate_dimension

        self.num_experts = config.num_experts
        self.top_k = config.top_k

        self.router = nn.Linear(self.embedding_dim, self.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [FeedForward(config) for _ in range(self.num_experts)]
        )

        self.config = config

    def forward(self, x):
        # A more intuitive approach for this logic can be applied using
        # traditional for-loops but we gain performance boosts when using
        # native Pytorch tensor ops.
        # This logic comes from here:
        #   https://github.com/dzhulgakov/llama-mistral/blob/main/llama/model.py#L350
        # Comments and explanations are my own

        orig_shape = x.shape
        # (B, T, C) -> (B*T, C)
        x = x.view(-1, self.embedding_dim)

        # (B*T, C) -> (B*T, num_experts)
        routing_logits = self.router(x)

        # (B*T, num_experts) -> (B*T, top_k)
        expert_logits, expert_indices = routing_logits.topk(self.top_k, dim=-1)
        expert_probs = expert_logits.softmax(dim=-1)
        expert_indices = expert_indices.view(-1)

        # Choosing the top K experts will gives us the logits and indices
        # of each expert for a given token. For each token in x we have to
        # pass it to every K experts, so we make our lives easier by repeating
        # each token in x K times.
        # Example:
        #   x with only two tokens = [[1.89,0.29,1.01], [0.2, 0.67, 1.2]]
        #   top K experts = [1, 2]
        #   repeated x K times = [
        #       [1.89,0.29,1.01], -> this will go to expert[1]
        #       [1.89,0.29,1.01], -> this will go to expert[1]
        #       [0.2, 0.67, 1.2], -> this will go to expert[2]
        #       [0.2, 0.67, 1.2]] -> this will go to expert[2]
        # (B*T, C) -> (B*T*top_k, C)
        x = x.repeat_interleave(self.top_k, dim=0)
        out = torch.empty_like(x)

        for i, expert in enumerate(self.experts):
            # Okay so this might be a bit confusing, but lets break it down:
            # x[expert_indices == i] finds the index of where i is located in
            # the expert_indices list. In other words, it finds which token
            # uses expert i. Once we find it, lets pass it through that expert's
            # linear layer and save the result in our out tensor.
            # Example:
            #   expert_indices = torch.tensor([1,2,3,4])
            #   x = torch.tensor([0.12, 0.94, 0.04, 0.55])
            #   i = 2
            #   x[expert_indices == i] -> x[[1,2,3,4] == 2] -> x[1] -> 0.94
            out[expert_indices == i] = expert(x[expert_indices == i])

        # (B*T*top_k, C) -> (B*T, top_k, C)
        out = out.view(*expert_probs.shape, -1)
        # (B*T, top_k) -> (B*T, top_k, 1)
        expert_probs = expert_probs.unsqueeze(-1)

        # So far we have only pass the tokens through each expert's linear layer
        # we still have to multiply it by their weights!
        # (B*T, top_k, C) -> (B*T, C)
        out = (out * expert_probs).sum(dim=1)

        # (B*T, C) -> (B, T, C)
        return out.view(*orig_shape)
