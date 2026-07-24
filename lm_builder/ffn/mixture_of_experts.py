from dataclasses import replace
import math

import torch
from torch import nn
from torch.nn import functional as F

from ..utils import is_positive_integer
from .config import FeedForwardConfig
from .feed_forward import FeedForward


class GroupedExperts(nn.Module):
    def __init__(self, config: FeedForwardConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        self.embedding_dim = config.embedding_dimension
        self.intermediate_dim = config.intermediate_dimension
        self.activation_fn = config.activation_fn()
        has_parameters = next(self.activation_fn.parameters(), None) is not None
        has_buffers = next(self.activation_fn.buffers(), None) is not None
        if has_parameters or has_buffers:
            raise ValueError(
                "MixtureOfExperts requires a stateless activation "
                "without parameters or buffers."
            )

        self.up_weight = nn.Parameter(
            torch.empty(
                self.num_experts,
                self.intermediate_dim,
                self.embedding_dim,
            )
        )
        self.gate_weight = nn.Parameter(
            torch.empty(
                self.num_experts,
                self.intermediate_dim,
                self.embedding_dim,
            )
        )
        self.down_weight = nn.Parameter(
            torch.empty(
                self.num_experts,
                self.embedding_dim,
                self.intermediate_dim,
            )
        )
        if config.bias:
            self.up_bias = nn.Parameter(
                torch.empty(self.num_experts, self.intermediate_dim)
            )
            self.gate_bias = nn.Parameter(
                torch.empty(self.num_experts, self.intermediate_dim)
            )
            self.down_bias = nn.Parameter(
                torch.empty(self.num_experts, self.embedding_dim)
            )
        else:
            self.register_parameter("up_bias", None)
            self.register_parameter("gate_bias", None)
            self.register_parameter("down_bias", None)
        self.dropout = nn.Dropout(config.dropout)

        self.reset_parameters(self.up_weight, self.up_bias, self.embedding_dim)
        self.reset_parameters(self.gate_weight, self.gate_bias, self.embedding_dim)
        self.reset_parameters(self.down_weight, self.down_bias, self.intermediate_dim)

    def reset_parameters(self, weight, bias, in_features):
        for expert_weight in weight:
            nn.init.kaiming_uniform_(expert_weight, a=math.sqrt(5))
        if bias is not None:
            bound = 1 / math.sqrt(in_features)
            nn.init.uniform_(bias, -bound, bound)

    def _run_expert(self, x, expert_index):
        up = F.linear(  # pylint: disable=not-callable
            x,
            self.up_weight[expert_index],
            None if self.up_bias is None else self.up_bias[expert_index],
        )
        gate = F.linear(  # pylint: disable=not-callable
            x,
            self.gate_weight[expert_index],
            None if self.gate_bias is None else self.gate_bias[expert_index],
        )
        hidden = up * self.activation_fn(gate)
        out = F.linear(  # pylint: disable=not-callable
            hidden,
            self.down_weight[expert_index],
            None if self.down_bias is None else self.down_bias[expert_index],
        )
        return self.dropout(out)

    def forward(
        self,
        x,
        flat_expert_weights,
        flat_expert_indices,
        expert_index,
    ):
        # Fetch this expert's flat assignment positions, then map them
        # back to token indices in the original input.
        # Example:
        #  Assume we have 3 tokens, top_k = 2, and 3 experts.
        #  If each token has assignments [[2, 0], [1, 2], [2, 1]],
        #  one possible assignment order is [1, 2, 5, 0, 3, 4].
        #  Equal expert IDs may appear in a different order without
        #  changing the grouping. Here expert 1's positions are [2, 5].
        #  Dividing by top_k and flooring maps them to token IDs [1, 2].
        
        # NOTE: We are doing sparse MoE routing by excluding the inactive experts,
        # rather than running all experts with 0 masking
        assignment_indices = torch.nonzero(
            flat_expert_indices == expert_index,
        ).flatten()
        token_indices = torch.div(
            assignment_indices,
            self.top_k,
            rounding_mode="floor",
        )

        # Choose the input tokens that are assigned to this expert and run them through the
        # expert's linear layer
        expert_inputs = x.index_select(0, token_indices)
        expert_output = self._run_expert(expert_inputs, expert_index)

        # Weight the expert's output by the routing weights for this expert
        routing_weights = flat_expert_weights.index_select(
            0,
            assignment_indices,
        )
        weighted_output = expert_output * routing_weights.unsqueeze(-1).to(
            expert_output.dtype
        )

        return token_indices, weighted_output.to(x.dtype)


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
        if not is_positive_integer(
            config.num_shared_experts,
            greater_than=-1,
        ):
            raise ValueError(
                "num_shared_experts must be a non-negative integer "
                "for MixtureOfExperts."
            )

    def __init__(self, config: FeedForwardConfig):
        super().__init__()
        self.validate_config(config)

        self.embedding_dim = config.embedding_dimension
        self.intermediate_dim = config.intermediate_dimension

        self.num_experts = config.num_experts
        self.top_k = config.top_k
        self.num_shared_experts = config.num_shared_experts

        self.router = nn.Linear(self.embedding_dim, self.num_experts, bias=False)
        self.experts = GroupedExperts(config)

        self.shared_expert = None
        if self.num_shared_experts:
            shared_expert_config = replace(
                config,
                intermediate_dimension=(
                    self.intermediate_dim * self.num_shared_experts
                ),
            )
            self.shared_expert = FeedForward(shared_expert_config)

        self.config = config

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

    def _execute_routed_experts(self, x, expert_weights, expert_indices):
        flat_expert_indices = expert_indices.reshape(-1)
        flat_expert_weights = expert_weights.reshape(-1)
        routed_token_indices = []
        weighted_outputs = []

        for expert_index in range(self.num_experts):
            token_indices, weighted_output = self.experts(
                x,
                flat_expert_weights,
                flat_expert_indices,
                expert_index,
            )

            # We have to keep track of which token indices had an expert assigned to
            # so that we can later sum all expert contributions for a given token
            routed_token_indices.append(token_indices)
            weighted_outputs.append(weighted_output)

        return torch.zeros_like(x).index_add(
            0,
            torch.cat(routed_token_indices),
            torch.cat(weighted_outputs),
        )

    def forward(self, x, token_mask=None):
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

        expert_weights = expert_logits.softmax(dim=-1)
        out = self._execute_routed_experts(
            x,
            expert_weights,
            expert_indices,
        )
        if self.shared_expert is not None:
            shared_out, _ = self.shared_expert(x)
            out = out + shared_out.to(out.dtype)

        # (B*T, C) -> (B, T, C)
        return out.reshape(*orig_shape), routing_loss
