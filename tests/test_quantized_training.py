"""CUDA numerical and saved-storage checks for the QT kernel prototype."""

import importlib.util
import os

import pytest
import torch

from dev.benchmarks.quantized_training import IntegerLinear, dependencies


def test_integer_training_gradients_and_saved_storage():
    if importlib.util.find_spec("torchao") is None:
        pytest.skip("Install mini_trainer[quantization]")
    if os.environ.get("RUN_CUDA_TESTS") != "1":
        pytest.skip("Set RUN_CUDA_TESTS=1 for the native INT8 training kernel test")
    if not torch.cuda.is_available():
        pytest.fail("CUDA requested but unavailable")
    torch.manual_seed(42)
    weight_type, _, _ = dependencies()
    inputs = torch.randn(32, 64, device="cuda", dtype=torch.float16, requires_grad=True)
    original_weight = torch.randn(64, 64, device="cuda", dtype=torch.float16) / 8
    weight = torch.nn.Parameter(weight_type.from_float(original_weight))
    grad_output = torch.randn(32, 64, device="cuda", dtype=torch.float16)
    saved = []

    def record(tensor):
        saved.append((tensor.dtype, tuple(tensor.shape)))
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(record, lambda tensor: tensor):
        output = IntegerLinear.apply(inputs, weight)
        output.backward(grad_output)
    assert (torch.int8, (32, 64)) in saved and (torch.int8, (64, 64)) in saved
    assert not any(dtype.is_floating_point and len(shape) == 2 for dtype, shape in saved)
    expected_output = inputs.detach().float() @ original_weight.float().T
    expected_dx = grad_output.float() @ original_weight.float()
    expected_dw = grad_output.float().T @ inputs.detach().float()
    for actual, expected in ((output, expected_output), (inputs.grad, expected_dx), (weight.grad, expected_dw)):
        relative_error = (actual.float() - expected).norm() / expected.norm()
        assert relative_error < 0.06  # Quantized arithmetic is approximate, not FP parity.
    before = weight.int_data.clone()
    optimizer = torch.optim.SGD([weight], lr=0.1, foreach=False)
    optimizer.step()
    assert not torch.equal(weight.int_data, before)
    assert weight.int_data.dtype == torch.int8
