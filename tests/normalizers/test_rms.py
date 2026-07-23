import torch

from lm_builder.normalizers import RMSNorm


def test_rms_norm_computes_statistics_in_float32():
    inputs = torch.tensor(
        [[1000.0, -1000.0, 500.0, -500.0]],
        dtype=torch.float16,
    )
    norm = RMSNorm(inputs.size(-1)).to(dtype=inputs.dtype)

    output = norm(inputs)

    float_inputs = inputs.float()
    expected = float_inputs * torch.rsqrt(
        float_inputs.pow(2).mean(dim=-1, keepdim=True) + norm.eps
    )
    expected = expected.to(inputs.dtype)

    assert output.dtype == inputs.dtype
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
