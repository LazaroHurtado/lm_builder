import copy

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from lm_builder.attention import (
    AttentionConfig,
    AttentionLayerConfig,
    CausalMultiHeadAttention,
)
from lm_builder.ffn import FeedForwardConfig, MixtureOfExperts
from lm_builder.transformer import Transformer, TransformerConfig


def capture_dynamic_output_shapes(test):
    # pylint: disable-next=protected-access,no-member
    config_patch = torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True)
    return config_patch(test)


# pylint: disable-next=too-many-arguments,too-many-positional-arguments
def build_moe(
    num_experts=4,
    top_k=2,
    dropout=0.0,
    num_shared_experts=0,
    bias=False,
    activation_fn=nn.GELU,
    dtype=torch.float32,
):
    model = MixtureOfExperts(
        FeedForwardConfig(
            embedding_dimension=8,
            intermediate_dimension=16,
            ffn_type=MixtureOfExperts,
            num_experts=num_experts,
            top_k=top_k,
            dropout=dropout,
            num_shared_experts=num_shared_experts,
            bias=bias,
            activation_fn=activation_fn,
        )
    )
    return model.to(dtype=dtype)


def build_moe_transformer(top_k=1):
    return Transformer(
        TransformerConfig(
            embedding_dimension=8,
            context_length=4,
            attention_config=AttentionConfig(
                qk_positional_embedding=None,
                layers=[
                    AttentionLayerConfig(
                        context_length=4,
                        embedding_dimension=8,
                        num_heads=2,
                        attention_type=CausalMultiHeadAttention,
                    )
                ],
            ),
            ffn_config=FeedForwardConfig(
                embedding_dimension=8,
                intermediate_dimension=16,
                ffn_type=MixtureOfExperts,
                num_experts=4,
                top_k=top_k,
            ),
            vocab_size=16,
            num_layers=1,
        )
    )


def run_expert(experts, expert_index, inputs):
    up = F.linear(  # pylint: disable=not-callable
        inputs,
        experts.up_weight[expert_index],
        None if experts.up_bias is None else experts.up_bias[expert_index],
    )
    gate = F.linear(  # pylint: disable=not-callable
        inputs,
        experts.gate_weight[expert_index],
        (None if experts.gate_bias is None else experts.gate_bias[expert_index]),
    )
    hidden = up * experts.activation_fn(gate)
    return F.linear(  # pylint: disable=not-callable
        hidden,
        experts.down_weight[expert_index],
        (None if experts.down_bias is None else experts.down_bias[expert_index]),
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
                run_expert(model.experts, expert_index, token)
                for expert_index in token_experts
            ]
        )
        outputs.append((expert_outputs * token_weights.unsqueeze(-1)).sum(dim=0))
    outputs = torch.stack(outputs)
    if model.shared_expert is not None:
        shared_output, _ = model.shared_expert(inputs)
        outputs = outputs + shared_output
    return outputs.reshape(original_shape)


def test_top_one_is_supported():
    model = build_moe(top_k=1)

    assert model.top_k == 1
    assert model.shared_expert is None


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


@pytest.mark.parametrize("num_shared_experts", [-1, 1.5, True])
def test_num_shared_experts_must_be_a_non_negative_integer(num_shared_experts):
    with pytest.raises(
        ValueError,
        match="num_shared_experts must be a non-negative integer",
    ):
        build_moe(num_shared_experts=num_shared_experts)


def test_shared_expert_uses_combined_intermediate_width():
    model = build_moe(num_shared_experts=2)

    assert model.num_shared_experts == 2
    assert model.shared_expert.hidden_dim == 2 * model.intermediate_dim


def test_shared_experts_build_from_config():
    config = FeedForwardConfig.build_config(
        {
            "type": "MixtureOfExperts",
            "intermediate_dimension": 16,
            "num_experts": 4,
            "top_k": 2,
            "num_shared_experts": 2,
        },
        embedding_dimension=8,
    )

    model = MixtureOfExperts(config)

    assert model.num_shared_experts == 2
    assert model.shared_expert.hidden_dim == 32


def test_stacked_expert_parameters_have_an_expert_dimension():
    model = build_moe(num_experts=4, bias=True)

    assert model.experts.up_weight.shape == (4, 16, 8)
    assert model.experts.gate_weight.shape == (4, 16, 8)
    assert model.experts.down_weight.shape == (4, 8, 16)
    assert model.experts.up_bias.shape == (4, 16)
    assert model.experts.down_bias.shape == (4, 8)


@pytest.mark.parametrize(
    ("embedding_dimension", "intermediate_dimension"),
    [
        (7, 16),
        (8, 15),
    ],
)
def test_expert_dimensions_do_not_require_kernel_alignment(
    embedding_dimension,
    intermediate_dimension,
):
    model = MixtureOfExperts(
        FeedForwardConfig(
            embedding_dimension=embedding_dimension,
            intermediate_dimension=intermediate_dimension,
            ffn_type=MixtureOfExperts,
            num_experts=4,
            top_k=2,
        )
    )
    inputs = torch.randn(2, 3, embedding_dimension)

    output, _ = model(inputs)

    assert output.shape == inputs.shape


def test_moe_rejects_stateful_activations():
    with pytest.raises(
        ValueError,
        match="requires a stateless activation",
    ):
        build_moe(activation_fn=nn.PReLU)


@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.float16, torch.bfloat16],
)
def test_floating_point_dtypes_are_supported(dtype):
    model = build_moe(dtype=dtype)
    inputs = torch.randn(2, 3, model.embedding_dim, dtype=dtype)

    output, _ = model(inputs)

    assert output.dtype == dtype
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("num_shared_experts", [0, 1, 2])
@pytest.mark.parametrize("top_k", [1, 2, 3, 4])
def test_forward_and_gradients_match_reference(top_k, num_shared_experts):
    torch.manual_seed(7)
    model = build_moe(
        top_k=top_k,
        num_shared_experts=num_shared_experts,
    )
    reference_model = copy.deepcopy(model)
    inputs = torch.randn(
        2,
        3,
        8,
        requires_grad=True,
    )
    reference_inputs = inputs.detach().clone().requires_grad_(True)

    output, routing_loss = model(inputs)
    reference_output = reference_forward(reference_model, reference_inputs)
    output.square().sum().backward()
    reference_output.square().sum().backward()

    assert torch.allclose(output, reference_output, atol=1e-5, rtol=1e-5)
    assert routing_loss is not None
    assert torch.allclose(
        inputs.grad,
        reference_inputs.grad,
        atol=1e-5,
        rtol=1e-5,
    )
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
                rtol=1e-5,
            )
    if model.shared_expert is not None:
        assert all(
            parameter.grad is not None for parameter in model.shared_expert.parameters()
        )


def test_bias_matches_reference():
    torch.manual_seed(13)
    model = build_moe(bias=True)
    inputs = torch.randn(2, 3, 8)

    output, _ = model(inputs)
    expected = reference_forward(model, inputs)

    assert torch.allclose(output, expected, atol=1e-5, rtol=1e-5)


def test_shared_expert_does_not_change_routing_loss():
    torch.manual_seed(11)
    routed_model = build_moe(top_k=1)
    torch.manual_seed(11)
    shared_model = build_moe(top_k=1, num_shared_experts=2)
    inputs = torch.randn(
        2,
        3,
        routed_model.embedding_dim,
    )

    _, routed_loss = routed_model(inputs)
    _, shared_loss = shared_model(inputs)

    assert torch.allclose(routed_loss, shared_loss)


def test_routing_loss_is_only_computed_during_training():
    model = build_moe(top_k=1)
    inputs = torch.randn(
        2,
        3,
        model.embedding_dim,
    )

    _, training_routing_loss = model(inputs)
    model.eval()
    _, eval_routing_loss = model(inputs)

    assert training_routing_loss is not None
    assert eval_routing_loss is None


def test_only_top_k_assignments_are_sent_to_experts(monkeypatch):
    model = build_moe()
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

    assignments_per_expert = []
    original_run_expert = model.experts._run_expert  # pylint: disable=protected-access

    def record_assignments(expert_inputs, expert_index):
        assignments_per_expert.append((expert_index, expert_inputs.size(0)))
        return original_run_expert(expert_inputs, expert_index)

    monkeypatch.setattr(model.experts, "_run_expert", record_assignments)
    model(torch.ones(1, 3, model.embedding_dim))

    assert assignments_per_expert == [(0, 3), (1, 3), (2, 0), (3, 0)]
    assert sum(count for _, count in assignments_per_expert) == 3 * model.top_k


def test_non_contiguous_input_matches_contiguous_input():
    torch.manual_seed(3)
    model = build_moe(num_shared_experts=1).eval()
    inputs = torch.randn(2, 8, 3).transpose(1, 2)

    assert not inputs.is_contiguous()
    assert torch.allclose(
        model(inputs)[0],
        model(inputs.contiguous())[0],
        atol=1e-6,
        rtol=1e-6,
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


@pytest.mark.parametrize("num_shared_experts", [0, 1])
def test_empty_input_preserves_shape(num_shared_experts):
    model = build_moe(num_shared_experts=num_shared_experts)
    inputs = torch.empty(
        2,
        0,
        model.embedding_dim,
        requires_grad=True,
    )

    output, routing_loss = model(inputs)

    assert output.shape == inputs.shape
    assert routing_loss == 0
    output.sum().backward()
    assert inputs.grad is not None


def test_expert_dropout_is_applied_only_during_training():
    model = build_moe(
        num_experts=2,
        dropout=1.0,
        num_shared_experts=1,
    )
    for weight in (
        model.experts.up_weight,
        model.experts.gate_weight,
        model.experts.down_weight,
        model.shared_expert.up_proj.weight,
        model.shared_expert.gate_proj.weight,
        model.shared_expert.down_proj.weight,
    ):
        nn.init.constant_(weight, 0.1)
    inputs = torch.ones(
        2,
        3,
        model.embedding_dim,
    )

    model.train()
    training_output, _ = model(inputs)
    model.eval()
    inference_output, _ = model(inputs)

    assert torch.count_nonzero(training_output) == 0
    assert torch.count_nonzero(inference_output) == inference_output.numel()


@capture_dynamic_output_shapes
def test_cpu_autocast_preserves_output_and_router_gradients():
    torch.manual_seed(5)
    model = build_moe(num_shared_experts=1, dtype=torch.float32)
    inputs = torch.randn(2, 3, model.embedding_dim, requires_grad=True)
    compiled = torch.compile(model, backend="eager", fullgraph=True)

    with torch.autocast("cpu", dtype=torch.bfloat16):
        output, routing_loss = compiled(inputs)
        loss = output.float().square().mean() + routing_loss
    loss.backward()

    assert output.dtype == inputs.dtype
    assert torch.isfinite(output).all()
    assert routing_loss.dtype == torch.float32
    assert model.router.weight.grad is not None
    assert torch.count_nonzero(model.router.weight.grad) > 0
    assert model.experts.up_weight.grad is not None
    assert torch.count_nonzero(model.experts.up_weight.grad) > 0


@capture_dynamic_output_shapes
def test_fullgraph_forward_and_backward_match_eager():
    torch.manual_seed(17)
    model = build_moe(top_k=2, num_shared_experts=1)
    inputs = torch.randn(
        2,
        3,
        model.embedding_dim,
    )
    token_mask = torch.tensor([[1, 1, 1], [0, 1, 1]])

    expected_output, expected_loss = model(inputs, token_mask=token_mask)
    compiled = torch.compile(model, backend="eager", fullgraph=True)
    output, routing_loss = compiled(inputs, token_mask=token_mask)
    (output.float().square().mean() + routing_loss).backward()

    torch.testing.assert_close(output, expected_output)
    torch.testing.assert_close(routing_loss, expected_loss)
    assert model.experts.up_weight.grad is not None
    assert model.router.weight.grad is not None


@capture_dynamic_output_shapes
def test_fullgraph_is_reused_when_routing_decisions_change():
    model = build_moe(top_k=2).eval()
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

    graph_count = 0

    def counting_backend(graph_module, _example_inputs):
        nonlocal graph_count
        graph_count += 1
        return graph_module.forward

    compiled = torch.compile(model, backend=counting_backend, fullgraph=True)
    positive_output, _ = compiled(torch.ones(1, 3, model.embedding_dim))
    negative_output, _ = compiled(-torch.ones(1, 3, model.embedding_dim))

    assert graph_count == 1
    assert not torch.equal(positive_output, negative_output)


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
    model = build_moe_transformer()
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    targets = torch.tensor([[2, 3, 4], [5, 6, 7]])

    _, cross_entropy_loss_without_targets, default_routing_loss = model(input_ids)
    logits, cross_entropy_loss, routing_loss = model(input_ids, targets)
    total_loss = cross_entropy_loss + 0.01 * routing_loss
    total_loss.backward()

    moe = model.blocks[0].ffn
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
    model = build_moe_transformer()
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


@capture_dynamic_output_shapes
def test_transformer_with_moe_compiles_fullgraph():
    model = build_moe_transformer(top_k=2).eval()
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

    expected = model(input_ids, attention_mask=attention_mask)
    compiled = torch.compile(model, backend="eager", fullgraph=True)
    actual = compiled(input_ids, attention_mask=attention_mask)

    torch.testing.assert_close(actual[0], expected[0])
    assert actual[1:] == expected[1:]


def run_compiled_device_test(device):
    torch.manual_seed(9)
    model = build_moe(top_k=1, num_shared_experts=1).to(device)
    inputs = torch.randn(
        2,
        3,
        model.embedding_dim,
        device=device,
        requires_grad=True,
    )

    with torch.no_grad():
        expected_output, expected_loss = model(inputs)
    compiled = torch.compile(model, fullgraph=True)
    output, routing_loss = compiled(inputs)
    (output.square().mean() + routing_loss).backward()

    torch.testing.assert_close(output, expected_output)
    torch.testing.assert_close(routing_loss, expected_loss)
    assert output.device.type == device
    assert torch.isfinite(output).all()
    assert routing_loss.device.type == device
    assert model.router.weight.grad is not None
    assert torch.count_nonzero(model.router.weight.grad) > 0
    assert model.experts.up_weight.grad is not None
    assert torch.count_nonzero(model.experts.up_weight.grad) > 0


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS is unavailable.",
)
@capture_dynamic_output_shapes
def test_mps_forward_and_backward():
    run_compiled_device_test("mps")


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is unavailable.",
)
@capture_dynamic_output_shapes
def test_cuda_forward_and_backward():
    run_compiled_device_test("cuda")
