"""Long-contraction integer arithmetic, including large-class input gradients."""

import os

import pytest
import torch

pytest.importorskip("torchao")
pytest.importorskip("triton")


def cuda():
    if os.environ.get("RUN_CUDA_TESTS") != "1":
        pytest.skip("Set RUN_CUDA_TESTS=1 for INT8 accumulator validation")
    if not torch.cuda.is_available():
        pytest.fail("CUDA requested but unavailable")
    return "cuda:0"


@pytest.mark.parametrize("contraction", [131071, 131072, 150000, 1000000])
def test_long_dot_matches_int64_reference(contraction):
    device = cuda()
    from mini_trainer.modeling._quantized_training import scaled_int8_mm

    left = torch.tensor([-128, 127], dtype=torch.int8, device=device)[:, None].expand(2, contraction).contiguous()
    right = torch.tensor([-128, 127, 1], dtype=torch.int8, device=device)[:, None].expand(3, contraction).contiguous().T
    row_scale = torch.tensor([0.01, 0.02], device=device)
    column_scale = torch.tensor([1.0, 0.5, -2.0], device=device)
    expected = (left.cpu().long() @ right.cpu().long()).float() * row_scale.cpu()[:, None] * column_scale.cpu()[None, :]
    actual = scaled_int8_mm(left, right, row_scale, column_scale)
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-6, atol=1e-4)


@pytest.mark.parametrize("compiled", [False, True])
def test_large_opposite_partial_sums_preserve_small_residual(compiled):
    device = cuda()
    from mini_trainer.modeling._quantized_training import scaled_int8_mm

    contraction = 262145
    left = torch.full((1, contraction), 127, dtype=torch.int8, device=device)
    # Make a large partial sum inexact in FP32; a float partial-sum workaround
    # loses one unit here even though the final integer result is small.
    left[0, 0] = 126
    right = torch.full((contraction, 1), 127, dtype=torch.int8, device=device)
    right[:131072].neg_()
    scale = torch.ones(1, device=device)
    function = torch.compile(scaled_int8_mm, fullgraph=True) if compiled else scaled_int8_mm
    actual = function(left, right, scale, scale)
    torch.testing.assert_close(actual, torch.full((1, 1), 127.0**2 + 127, device=device), rtol=0, atol=0)


def test_million_output_linear_input_gradient():
    device = cuda()
    from mini_trainer.modeling.quantized_training import prepare_quantized_training

    model = torch.nn.Linear(8, 1000000, bias=False, device=device)
    with torch.no_grad():
        model.weight.fill_(1)
    prepare_quantized_training(model)
    inputs = torch.ones(2, 8, device=device, requires_grad=True)
    model(inputs).sum().backward()
    torch.testing.assert_close(inputs.grad, torch.full_like(inputs, 1000000), rtol=1e-6, atol=1e-3)
    torch.testing.assert_close(model.weight.grad, torch.full_like(model.weight.grad, 2), rtol=1e-6, atol=1e-5)
