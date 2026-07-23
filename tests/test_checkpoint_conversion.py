from collections import OrderedDict

import torch

from examples.gpt2 import GPT2Loader
from examples.llama2 import Llama2Loader


def test_gpt2_checkpoint_keeps_fused_qkv_projection():
    qkv_weight = torch.arange(48).view(4, 12)
    qkv_bias = torch.arange(12)
    original_state_dict = OrderedDict(
        {
            "transformer.h.0.attn.c_attn.weight": qkv_weight,
            "transformer.h.0.attn.c_attn.bias": qkv_bias,
        }
    )

    state_dict = GPT2Loader.convert_state_dict(original_state_dict)

    assert list(state_dict) == [
        "blocks.0.attn.qkv_proj.weight",
        "blocks.0.attn.qkv_proj.bias",
    ]
    assert torch.equal(
        state_dict["blocks.0.attn.qkv_proj.weight"],
        qkv_weight.t(),
    )
    assert torch.equal(
        state_dict["blocks.0.attn.qkv_proj.bias"],
        qkv_bias,
    )


def test_llama_checkpoint_combines_qkv_projections():
    query_weight = torch.full((8, 8), 1.0)
    key_weight = torch.full((4, 8), 2.0)
    value_weight = torch.full((4, 8), 3.0)
    query_bias = torch.full((8,), 4.0)
    key_bias = torch.full((4,), 5.0)
    value_bias = torch.full((4,), 6.0)
    output_weight = torch.full((8, 8), 7.0)
    original_state_dict = OrderedDict(
        {
            "model.layers.0.self_attn.q_proj.weight": query_weight,
            "model.layers.0.self_attn.k_proj.weight": key_weight,
            "model.layers.0.self_attn.v_proj.weight": value_weight,
            "model.layers.0.self_attn.q_proj.bias": query_bias,
            "model.layers.0.self_attn.k_proj.bias": key_bias,
            "model.layers.0.self_attn.v_proj.bias": value_bias,
            "model.layers.0.self_attn.o_proj.weight": output_weight,
        }
    )

    state_dict = Llama2Loader().convert_state_dict(original_state_dict)

    assert torch.equal(
        state_dict["blocks.0.attn.qkv_proj.weight"],
        torch.cat((query_weight, key_weight, value_weight)),
    )
    assert torch.equal(
        state_dict["blocks.0.attn.qkv_proj.bias"],
        torch.cat((query_bias, key_bias, value_bias)),
    )
    assert torch.equal(
        state_dict["blocks.0.attn.out_proj.weight"],
        output_weight,
    )
    assert all(
        projection_name not in layer_name
        for layer_name in state_dict
        for projection_name in (".q_proj.", ".k_proj.", ".v_proj.")
    )
