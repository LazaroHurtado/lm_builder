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
                model.experts[expert_index.item()](token)
                for expert_index in token_experts
            ]
        )
        outputs.append((expert_outputs * token_weights.unsqueeze(-1)).sum(dim=0))
    return torch.stack(outputs).reshape(original_shape)


def test_top_one_requires_routing_loss():
    with pytest.raises(ValueError, match="top_k=1 without routing loss"):
        FeedForwardConfig(
            embedding_dimension=8,
            intermediate_dimension=16,
            ffn_type=MixtureOfExperts,
            num_experts=4,
            top_k=1,
        )


@pytest.mark.parametrize("num_experts", [0, -1, 1.5, True])
def test_num_experts_must_be_a_positive_integer(num_experts):
    with pytest.raises(ValueError, match="num_experts must be a positive integer"):
        FeedForwardConfig(
            embedding_dimension=8,
            intermediate_dimension=16,
            ffn_type=MixtureOfExperts,
            num_experts=num_experts,
            top_k=2,
        )


@pytest.mark.parametrize("top_k", [0, -1, 1.5, True, 5])
def test_top_k_must_be_between_two_and_num_experts(top_k):
    with pytest.raises(
        ValueError,
        match="top_k must be an integer between 2 and num_experts",
    ):
        FeedForwardConfig(
            embedding_dimension=8,
            intermediate_dimension=16,
            ffn_type=MixtureOfExperts,
            num_experts=4,
            top_k=top_k,
        )


@pytest.mark.parametrize("top_k", [2, 3, 4])
def test_forward_and_gradients_match_reference(top_k):
    torch.manual_seed(7)
    model = build_moe(top_k=top_k)
    reference_model = copy.deepcopy(model)
    inputs = torch.randn(2, 3, 8, requires_grad=True)
    reference_inputs = inputs.detach().clone().requires_grad_(True)

    output = model(inputs)
    reference_output = reference_forward(reference_model, reference_inputs)
    output.square().sum().backward()
    reference_output.square().sum().backward()

    assert torch.allclose(output, reference_output, atol=1e-6)
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


def test_only_active_experts_are_called():
    class CountingExpert(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, inputs):
            self.calls += 1
            return inputs

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
        model(inputs),
        model(inputs.contiguous()),
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

    output = model(inputs)

    assert output.shape == inputs.shape
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
    training_output = model(inputs)
    model.eval()
    inference_output = model(inputs)

    assert torch.count_nonzero(training_output) == 0
    assert torch.count_nonzero(inference_output) == inference_output.numel()


def test_cpu_autocast_preserves_output_and_router_gradients():
    torch.manual_seed(5)
    model = build_moe()
    inputs = torch.randn(2, 3, model.embedding_dim, requires_grad=True)

    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = model(inputs)
        loss = output.float().square().mean()
    loss.backward()

    assert output.dtype == inputs.dtype
    assert torch.isfinite(output).all()
    assert model.router.weight.grad is not None
    assert torch.count_nonzero(model.router.weight.grad) > 0


def test_transformer_uses_moe_and_propagates_router_gradients():
    ffn_config = FeedForwardConfig(
        embedding_dimension=8,
        intermediate_dimension=16,
        ffn_type=MixtureOfExperts,
        num_experts=4,
        top_k=2,
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

    logits, loss = model(input_ids, targets)
    loss.backward()

    moe = model.transformer.blocks[0].ffn
    assert logits.shape == (2, 3, 16)
    assert isinstance(moe, MixtureOfExperts)
    assert moe.router.weight.grad is not None
    assert torch.count_nonzero(moe.router.weight.grad) > 0


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS is unavailable.",
)
def test_mps_forward_and_backward():
    torch.manual_seed(9)
    model = build_moe().to("mps")
    inputs = torch.randn(
        2,
        3,
        model.embedding_dim,
        device="mps",
        requires_grad=True,
    )

    output = model(inputs)
    output.square().mean().backward()

    assert output.device.type == "mps"
    assert torch.isfinite(output).all()
    assert model.router.weight.grad is not None
    assert torch.count_nonzero(model.router.weight.grad) > 0
