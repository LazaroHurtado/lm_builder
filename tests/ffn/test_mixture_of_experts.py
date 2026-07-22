import copy

import pytest
import torch
from torch import nn

from lm_builder.attention import AttentionConfig, CausalMultiHeadAttention
from lm_builder.ffn import FeedForwardConfig, MixtureOfExperts
from lm_builder.transformer import Transformer, TransformerConfig


def build_moe(num_experts=4, top_k=2, dropout=0.0):
    return MixtureOfExperts(
        FeedForwardConfig(
            embedding_dimension=8,
            intermediate_dimension=16,
            ffn_type=MixtureOfExperts,
            num_experts=num_experts,
            top_k=top_k,
            dropout=dropout,
        )
    )


def reference_forward(model, inputs):
    original_shape = inputs.shape
    inputs = inputs.reshape(-1, model.embedding_dim)
    selected_logits, selected_experts = model.router(inputs).topk(
        model.top_k,
        dim=-1,
    )
    routing_weights = selected_logits.softmax(dim=-1)
    outputs = []
    for token, token_experts, token_weights in zip(
        inputs,
        selected_experts,
        routing_weights,
    ):
        expert_outputs = torch.stack(
            [
                model.experts[expert_index.item()](token)[0]
                for expert_index in token_experts
            ]
        )
        outputs.append((expert_outputs * token_weights.unsqueeze(-1)).sum(dim=0))
    return torch.stack(outputs).reshape(original_shape)


def test_top_one_is_supported():
    model = build_moe(top_k=1)

    assert model.top_k == 1


@pytest.mark.parametrize("num_experts", [0, -1, 1.5, True])
def test_num_experts_must_be_a_positive_integer(num_experts):
    with pytest.raises(ValueError, match="num_experts must be a positive integer"):
        MixtureOfExperts(
            FeedForwardConfig(
                embedding_dimension=8,
                intermediate_dimension=16,
                ffn_type=MixtureOfExperts,
                num_experts=num_experts,
                top_k=2,
            )
        )


@pytest.mark.parametrize("top_k", [0, -1, 1.5, True, 5])
def test_top_k_must_be_between_one_and_num_experts(top_k):
    with pytest.raises(
        ValueError,
        match="top_k must be an integer between 1 and num_experts",
    ):
        MixtureOfExperts(
            FeedForwardConfig(
                embedding_dimension=8,
                intermediate_dimension=16,
                ffn_type=MixtureOfExperts,
                num_experts=4,
                top_k=top_k,
            )
        )


@pytest.mark.parametrize("top_k", [1, 2, 3, 4])
def test_forward_and_gradients_match_reference(top_k):
    torch.manual_seed(7)
    model = build_moe(top_k=top_k)
    reference_model = copy.deepcopy(model)
    inputs = torch.randn(2, 3, 8, requires_grad=True)
    reference_inputs = inputs.detach().clone().requires_grad_(True)

    output, routing_loss = model(inputs)
    reference_output = reference_forward(reference_model, reference_inputs)
    output.square().sum().backward()
    reference_output.square().sum().backward()

    assert torch.allclose(output, reference_output, atol=1e-6)
    assert routing_loss is not None
    assert torch.allclose(inputs.grad, reference_inputs.grad, atol=1e-6)
    for parameter, reference_parameter in zip(
        model.parameters(),
        reference_model.parameters(),
    ):
        if parameter.grad is None or reference_parameter.grad is None:
            assert parameter.grad is reference_parameter.grad
        else:
            assert torch.allclose(
                parameter.grad,
                reference_parameter.grad,
                atol=1e-5,
            )


def test_routing_loss_is_only_computed_during_training():
    model = build_moe(top_k=1)
    inputs = torch.randn(2, 3, model.embedding_dim)

    _, training_routing_loss = model(inputs)
    model.eval()
    _, eval_routing_loss = model(inputs)

    assert training_routing_loss is not None
    assert eval_routing_loss is None


def test_only_active_experts_are_called():
    class CountingExpert(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, inputs):
            self.calls += 1
            return inputs, None

    model = build_moe()
    model.experts = nn.ModuleList([CountingExpert() for _ in model.experts])
    with torch.no_grad():
        model.router.weight.copy_(
            torch.tensor(
                [
                    [2.0] * model.embedding_dim,
                    [1.0] * model.embedding_dim,
                    [-1.0] * model.embedding_dim,
                    [-2.0] * model.embedding_dim,
                ]
            )
        )

    model(torch.ones(1, 3, model.embedding_dim))

    assert [expert.calls for expert in model.experts] == [1, 1, 0, 0]


def test_non_contiguous_input_matches_contiguous_input():
    torch.manual_seed(3)
    model = build_moe().eval()
    inputs = torch.randn(2, 8, 3).transpose(1, 2)

    assert not inputs.is_contiguous()
    assert torch.allclose(
        model(inputs)[0],
        model(inputs.contiguous())[0],
        atol=1e-6,
    )


def test_wrong_input_dimension_is_rejected():
    model = build_moe()

    with pytest.raises(
        ValueError,
        match="last dimension to be 8, but received 4",
    ):
        model(torch.randn(2, 3, 4))


def test_scalar_input_is_rejected():
    model = build_moe()

    with pytest.raises(ValueError, match="at least one dimension"):
        model(torch.tensor(1.0))


def test_empty_input_preserves_shape():
    model = build_moe()
    inputs = torch.empty(2, 0, model.embedding_dim, requires_grad=True)

    output, routing_loss = model(inputs)

    assert output.shape == inputs.shape
    assert routing_loss == 0
    output.sum().backward()
    assert inputs.grad is not None


def test_expert_dropout_is_applied_only_during_training():
    model = build_moe(num_experts=2, dropout=1.0)
    for expert in model.experts:
        for projection in (
            expert.up_proj,
            expert.gate_proj,
            expert.down_proj,
        ):
            nn.init.constant_(projection.weight, 0.1)
    inputs = torch.ones(2, 3, model.embedding_dim)

    model.train()
    training_output, _ = model(inputs)
    model.eval()
    inference_output, _ = model(inputs)

    assert torch.count_nonzero(training_output) == 0
    assert torch.count_nonzero(inference_output) == inference_output.numel()


def test_cpu_autocast_preserves_output_and_router_gradients():
    torch.manual_seed(5)
    model = build_moe()
    inputs = torch.randn(2, 3, model.embedding_dim, requires_grad=True)

    with torch.autocast("cpu", dtype=torch.bfloat16):
        output, routing_loss = model(inputs)
        loss = output.float().square().mean() + routing_loss
    loss.backward()

    assert output.dtype == inputs.dtype
    assert torch.isfinite(output).all()
    assert routing_loss.dtype == torch.float32
    assert model.router.weight.grad is not None
    assert torch.count_nonzero(model.router.weight.grad) > 0


def test_routing_loss_penalizes_collapsed_experts():
    model = build_moe(num_experts=2, top_k=1)
    balanced_logits = torch.tensor([[8.0, -8.0], [-8.0, 8.0]])
    collapsed_logits = torch.tensor([[8.0, -8.0], [8.0, -8.0]])

    balanced_loss = model._get_routing_loss(  # pylint: disable=protected-access
        balanced_logits,
        balanced_logits.topk(1, dim=-1).indices,
    )
    collapsed_loss = model._get_routing_loss(  # pylint: disable=protected-access
        collapsed_logits,
        collapsed_logits.topk(1, dim=-1).indices,
    )

    assert torch.allclose(balanced_loss, torch.tensor(1.0), atol=1e-5)
    assert collapsed_loss > balanced_loss


def test_routing_loss_ignores_masked_tokens():
    model = build_moe(num_experts=2, top_k=1)
    routing_logits = torch.tensor([[8.0, -8.0], [-8.0, 8.0], [8.0, -8.0]])
    expert_indices = routing_logits.topk(1, dim=-1).indices

    routing_loss = model._get_routing_loss(  # pylint: disable=protected-access
        routing_logits,
        expert_indices,
        token_mask=torch.tensor([[1, 1, 0]]),
    )

    assert torch.allclose(routing_loss, torch.tensor(1.0), atol=1e-5)


def test_transformer_returns_top_one_routing_loss():
    ffn_config = FeedForwardConfig(
        embedding_dimension=8,
        intermediate_dimension=16,
        ffn_type=MixtureOfExperts,
        num_experts=4,
        top_k=1,
    )
    model = Transformer(
        TransformerConfig(
            embedding_dimension=8,
            context_length=4,
            attention_config=[
                AttentionConfig(
                    context_length=4,
                    embedding_dimension=8,
                    num_heads=2,
                    attention_type=CausalMultiHeadAttention,
                )
            ],
            ffn_config=ffn_config,
            vocab_size=16,
            num_layers=1,
        )
    )
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    targets = torch.tensor([[2, 3, 4], [5, 6, 7]])

    _, cross_entropy_loss_without_targets, default_routing_loss = model(input_ids)
    logits, cross_entropy_loss, routing_loss = model(input_ids, targets)
    total_loss = cross_entropy_loss + 0.01 * routing_loss
    total_loss.backward()

    moe = model.transformer.blocks[0].ffn
    assert logits.shape == (2, 3, 16)
    assert isinstance(moe, MixtureOfExperts)
    assert cross_entropy_loss_without_targets is None
    assert default_routing_loss is not None
    assert cross_entropy_loss is not None
    assert routing_loss is not None
    assert routing_loss.ndim == 0
    assert moe.router.weight.grad is not None
    assert torch.count_nonzero(moe.router.weight.grad) > 0


def test_transformer_routing_loss_ignores_left_padding():
    model = Transformer(
        TransformerConfig(
            embedding_dimension=8,
            context_length=4,
            attention_config=[
                AttentionConfig(
                    context_length=4,
                    embedding_dimension=8,
                    num_heads=2,
                    attention_type=CausalMultiHeadAttention,
                )
            ],
            ffn_config=FeedForwardConfig(
                embedding_dimension=8,
                intermediate_dimension=16,
                ffn_type=MixtureOfExperts,
                num_experts=4,
                top_k=1,
            ),
            vocab_size=16,
            num_layers=1,
        )
    )
    input_ids = torch.tensor([[2, 3]])
    attention_mask = torch.tensor([[1, 1]])
    targets = torch.tensor([[3, 4]])
    padded_input_ids = torch.tensor([[0, 0, 2, 3]])
    padded_attention_mask = torch.tensor([[0, 0, 1, 1]])
    padded_targets = torch.tensor([[-1, -1, 3, 4]])

    with torch.no_grad():
        _, _, routing_loss = model(
            input_ids,
            targets,
            attention_mask=attention_mask,
        )
        _, _, padded_routing_loss = model(
            padded_input_ids,
            padded_targets,
            attention_mask=padded_attention_mask,
        )

    assert torch.allclose(routing_loss, padded_routing_loss, atol=1e-6)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS is unavailable.",
)
def test_mps_forward_and_backward():
    torch.manual_seed(9)
    model = build_moe(top_k=1).to("mps")
    inputs = torch.randn(
        2,
        3,
        model.embedding_dim,
        device="mps",
        requires_grad=True,
    )

    output, routing_loss = model(inputs)
    (output.square().mean() + routing_loss).backward()

    assert output.device.type == "mps"
    assert torch.isfinite(output).all()
    assert routing_loss.device.type == "mps"
    assert model.router.weight.grad is not None
    assert torch.count_nonzero(model.router.weight.grad) > 0
