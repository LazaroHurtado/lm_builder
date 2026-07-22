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
                "top_k must be an integer between 1 and num_experts "
                "for MixtureOfExperts."
            )
        if config.top_k > config.num_experts:
            raise ValueError(
                "top_k must be an integer between 1 and num_experts "
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
            expert_out, _ = expert(x)
            expert_out = expert_out.to(out.dtype)
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

    def _get_routing_loss(
        self,
        routing_logits,
        expert_indices,
        token_mask=None,
    ):
        if not self.training:
            return None

        router_probs = routing_logits.softmax(dim=-1, dtype=torch.float32)
        if token_mask is not None:
            token_mask = token_mask.reshape(-1).to(
                device=routing_logits.device,
                dtype=router_probs.dtype,
            )
            if token_mask.numel() != routing_logits.size(0):
                raise ValueError(
                    "Routing token mask must match the number of input tokens."
                )

        if routing_logits.size(0) == 0:
            return routing_logits.float().sum()

        if token_mask is None:
            assignment_weights = router_probs.new_ones(expert_indices.numel())
            valid_token_count = routing_logits.size(0)
            router_prob_per_expert = router_probs.mean(dim=0)
        else:
            valid_token_count = token_mask.sum().clamp_min(1.0)
            assignment_weights = (
                token_mask[:, None].expand(-1, expert_indices.size(1)).reshape(-1)
            )
            router_prob_per_expert = (router_probs * token_mask[:, None]).sum(
                dim=0
            ) / valid_token_count

        tokens_per_expert = router_probs.new_zeros(self.num_experts)
        tokens_per_expert.index_add_(
            0,
            expert_indices.reshape(-1),
            assignment_weights,
        )
        tokens_per_expert = tokens_per_expert / valid_token_count

        # This is the main logic that measures how balanced our expert routing gate is.
        # We can see that this loss is minimized only when the router probabilities are uniform across all experts.
        # Re-written, this is saying N_e * \sum_{i=1}^{N_e} (T_i / T)*(P_i / T),
        # where N_e is the number of experts, T_i is the number of tokens assigned to expert i, T is the valid token count,
        # and P_i is the router probability for expert i.
        # Only when the routing is uniform will T_i = T / N_e and P_i = 1 / N_e, leading to
        # N_e * (N_e * (1 / N_e^2)) = 1, which is the minimum value for this loss.
        return self.num_experts * torch.sum(tokens_per_expert * router_prob_per_expert)

    def forward(  # pylint: disable=too-many-locals
        self,
        x,
        token_mask=None,  # token mask is needed to know which tokens are valid for routing loss calculation (padding tokens should not contribute to the routing loss)
    ):
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

        # (B*T, C) -> (B*T, num_experts)
        routing_logits = self.router(x)

        # The top-k routing logic comes from here:
        #   https://github.com/dzhulgakov/llama-mistral/blob/main/llama/model.py#L350
        # Comments and explanations are my own
        # (B*T, num_experts) -> (B*T, top_k)
        expert_logits, expert_indices = routing_logits.topk(self.top_k, dim=-1)
        routing_loss = self._get_routing_loss(
            routing_logits,
            expert_indices,
            token_mask,
        )
        if x.numel() == 0:
            return x.reshape(*orig_shape), routing_loss

        if self.top_k == self.num_experts:
            expert_probs = routing_logits.softmax(dim=-1)
            out = self._dispatch_all(x, expert_probs)
            # (B*T, C) -> (B, T, C)
            return out.reshape(*orig_shape), routing_loss

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

            expert_out, _ = expert(x.index_select(0, token_indices))
            expert_out = expert_out.to(out.dtype)
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
        return out.reshape(*orig_shape), routing_loss
