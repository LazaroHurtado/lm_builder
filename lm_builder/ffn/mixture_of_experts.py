import torch
from torch import nn

from ..utils import is_positive_integer
from .config import FeedForwardConfig
from .feed_forward import FeedForward


class MixtureOfExperts(nn.Module):

    @staticmethod
    def validate_config(config: FeedForwardConfig):
        if not is_positive_integer(config.num_experts):
            raise ValueError(
                "num_experts must be a positive integer for MixtureOfExperts."
            )
        if not is_positive_integer(config.top_k):
            raise ValueError(
                "top_k must be an integer between 2 and num_experts "
                "for MixtureOfExperts."
            )
        if config.top_k == 1:
            raise ValueError(
                "MixtureOfExperts does not support top_k=1 without routing loss."
            )
        if config.top_k > config.num_experts:
            raise ValueError(
                "top_k must be an integer between 2 and num_experts "
                "for MixtureOfExperts."
            )

    def __init__(self, config: FeedForwardConfig):
        super().__init__()
        self.validate_config(config)

        self.embedding_dim = config.embedding_dimension
        self.intermediate_dim = config.intermediate_dimension

        self.num_experts = config.num_experts
        self.top_k = config.top_k

        self.router = nn.Linear(self.embedding_dim, self.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [FeedForward(config) for _ in range(self.num_experts)]
        )

        self.config = config

    def _dispatch_all(self, x, expert_probs):
        out = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            expert_out = expert(x).to(out.dtype)
            expert_weights = expert_probs[:, i, None].to(out.dtype)
            out.add_(expert_out * expert_weights)

        return out

    def _get_token_expert_assignments(self, expert_indices):
        assignment_order = expert_indices.argsort()
        sorted_experts = expert_indices.index_select(0, assignment_order)

        # Get all the experts we will use and how many tokens are assigned
        # to each expert.
        active_experts, assignment_counts = torch.unique_consecutive(
            sorted_experts,
            return_counts=True,
        )

        expert_assignments = (
            torch.stack(
                (active_experts, assignment_counts),
                dim=-1,
            )
            .cpu()
            .tolist()
        )

        return expert_assignments, assignment_order

    def forward(self, x):  # pylint: disable=too-many-locals
        if x.ndim == 0:
            raise ValueError("MixtureOfExperts input must have at least one dimension.")
        if x.shape[-1] != self.embedding_dim:
            raise ValueError(
                "MixtureOfExperts expected the input's last dimension "
                f"to be {self.embedding_dim}, but received {x.shape[-1]}."
            )

        orig_shape = x.shape
        # (B, T, C) -> (B*T, C)
        x = x.reshape(-1, self.embedding_dim)
        if x.numel() == 0:
            return x.reshape(*orig_shape)

        # (B*T, C) -> (B*T, num_experts)
        routing_logits = self.router(x)
        if self.top_k == self.num_experts:
            expert_probs = routing_logits.softmax(dim=-1)
            out = self._dispatch_all(x, expert_probs)
            # (B*T, C) -> (B, T, C)
            return out.reshape(*orig_shape)

        # The top-k routing logic comes from here:
        #   https://github.com/dzhulgakov/llama-mistral/blob/main/llama/model.py#L350
        # Comments and explanations are my own
        # (B*T, num_experts) -> (B*T, top_k)
        expert_logits, expert_indices = routing_logits.topk(self.top_k, dim=-1)
        expert_probs = expert_logits.softmax(dim=-1).reshape(-1)
        expert_indices = expert_indices.reshape(-1)

        # Choosing the top K experts will gives us the logits and indices
        # of each expert for a given token. For each token in x we have to
        # pass it to every K experts.
        expert_assignments, assignment_order = self._get_token_expert_assignments(
            expert_indices
        )

        out = torch.zeros_like(x)
        assignment_offset = 0
        for i, assignment_count in expert_assignments:
            expert = self.experts[i]

            # Fetch this expert's flat assignment positions, then map them
            # back to token indices in the original input.
            # Example:
            #  Assume we have 3 tokens, top_k = 2, and 3 experts.
            #  If each token has assignments [[2, 0], [1, 2], [2, 1]],
            #  one possible assignment order is [1, 2, 5, 0, 3, 4].
            #  Equal expert IDs may appear in a different order without
            #  changing the grouping. Here expert 1's positions are [2, 5].
            #  Dividing by top_k and flooring maps them to token IDs [1, 2].
            assignments = assignment_order.narrow(
                0,
                assignment_offset,
                assignment_count,
            )
            token_indices = torch.div(
                assignments,
                self.top_k,
                rounding_mode="floor",
            )

            expert_out = expert(x.index_select(0, token_indices)).to(out.dtype)
            # So far we have only pass the tokens through each expert's linear layer
            # we still have to multiply it by their weights!
            expert_weights = (
                expert_probs.index_select(0, assignments).unsqueeze(-1).to(out.dtype)
            )
            out.index_add_(
                0,
                token_indices,
                expert_out * expert_weights,
            )

            assignment_offset += assignment_count

        # (B*T, C) -> (B, T, C)
        return out.reshape(*orig_shape)
