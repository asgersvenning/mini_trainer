"""CUDA numerical and saved-storage checks for the QT kernel prototype."""

import importlib.util
import os

import pytest
import torch

from dev.benchmarks.quantized_training import IntegerLinear, dependencies


@pytest.mark.parametrize("gradient_scale", [1.0, 1e-6])
def test_integer_training_gradients_and_saved_storage(gradient_scale):
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
    grad_output = torch.randn(32, 64, device="cuda", dtype=torch.float16) * gradient_scale
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
    optimizer = torch.optim.SGD([weight], lr=0.1 / gradient_scale, foreach=False)
    optimizer.step()
    assert not torch.equal(weight.int_data, before)
    assert weight.int_data.dtype == torch.int8


@pytest.mark.parametrize("optimizer_name", ["sgd", "adamw"])
@pytest.mark.parametrize("foreach", [None, False, True])
def test_integer_optimizer_updates_and_continuation(optimizer_name, foreach):
    """Compare update math before rounding and resume the stochastic trajectory."""
    import copy
    import io

    pytest.importorskip("torchao")
    from dev.benchmarks._int8_weight import TrainingWeight

    torch.manual_seed(123)
    parameter = torch.nn.Parameter(TrainingWeight.from_float(torch.randn(32, 64)))
    reference = torch.nn.Parameter(parameter.dequantize().clone())
    cls = torch.optim.SGD if optimizer_name == "sgd" else torch.optim.AdamW
    options = dict(lr=0.03, weight_decay=0.1, foreach=foreach)
    if optimizer_name == "sgd":
        options.update(momentum=0.9, nesterov=True)
    optimizer, reference_optimizer = cls([parameter], **options), cls([reference], **options)
    for _ in range(4):
        # Start from the same represented weights, keeping optimizer state.
        with torch.no_grad():
            reference.copy_(parameter.dequantize())
        gradient = torch.randn_like(reference)
        parameter.grad, reference.grad = gradient.clone(), gradient.clone()
        optimizer.step()
        reference_optimizer.step()
        error = (parameter.dequantize() - reference).abs()
        # One stochastic quantization adds less than one quantization interval
        # per element (plus scale calculation rounding).
        assert torch.all(error <= reference.detach().abs().amax(1, keepdim=True) / 127 + 1e-6)
    checkpoint = io.BytesIO()
    torch.save({"weight": parameter.detach(), "optimizer": optimizer.state_dict(), "rng": torch.get_rng_state()}, checkpoint)
    checkpoint.seek(0)
    with torch.serialization.safe_globals([TrainingWeight]):
        restored = torch.load(checkpoint, weights_only=True)
    resumed = torch.nn.Parameter(restored["weight"])
    resumed_optimizer = cls([resumed], **options)
    resumed_optimizer.load_state_dict(restored["optimizer"])
    gradient = torch.randn_like(reference)
    parameter.grad, resumed.grad = gradient.clone(), gradient.clone()
    torch.set_rng_state(restored["rng"])
    optimizer.step()
    torch.set_rng_state(restored["rng"])
    resumed_optimizer.step()
    assert torch.equal(parameter.int_data, resumed.int_data)
    assert torch.equal(parameter.scale, resumed.scale)
    assert isinstance(copy.deepcopy(parameter), TrainingWeight)
    assert isinstance(parameter.to(torch.float64), TrainingWeight)
    assert isinstance(parameter.detach(), TrainingWeight)
    for key, value in optimizer.state[parameter].items():
        other = resumed_optimizer.state[resumed][key]
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, other)
        else:
            assert value == other


def test_integer_decay_preserves_codes_and_upstream_dispatch():
    pytest.importorskip("torchao")
    from torchao.prototype.quantized_training.int8 import Int8QuantizedTrainingLinearWeight

    from dev.benchmarks._int8_weight import TrainingWeight

    original = torch.randn(8, 16)
    weight = TrainingWeight.from_float(original)
    codes, represented = weight.int_data.clone(), weight.dequantize()
    weight.mul_(0.9)
    assert torch.equal(weight.int_data, codes)
    torch.testing.assert_close(weight.dequantize(), represented * 0.9)
    upstream = Int8QuantizedTrainingLinearWeight.from_float(original)
    assert type(upstream.detach()) is Int8QuantizedTrainingLinearWeight
    with pytest.raises(NotImplementedError):
        upstream.mul_(0.9)


@pytest.mark.parametrize("dtype,epsilon", [(torch.float16, 1e-4), (torch.float32, 1e-8)])
def test_integer_linear_module_cuda(dtype, epsilon):
    pytest.importorskip("torchao")
    if os.environ.get("RUN_CUDA_TESTS") != "1":
        pytest.skip("Set RUN_CUDA_TESTS=1 for CUDA linear dispatch")
    if not torch.cuda.is_available():
        pytest.fail("CUDA requested but unavailable")
    from dev.benchmarks._int8_weight import TrainingWeight

    torch.manual_seed(456)
    layer = torch.nn.Linear(64, 32, device="cuda", dtype=dtype)
    layer.weight = torch.nn.Parameter(TrainingWeight.from_float(layer.weight))
    inputs = torch.randn(2, 16, 64, device="cuda", dtype=dtype, requires_grad=True)
    expected = torch.nn.functional.linear(inputs.float(), layer.weight.dequantize().float(), layer.bias.float())
    output = layer(inputs)
    assert output.shape == (2, 16, 32)
    assert (output.float() - expected).norm() / expected.norm() < 0.04
    output.float().square().mean().backward()
    assert torch.isfinite(inputs.grad).all()
    assert torch.isfinite(layer.bias.grad).all()
    optimizer = torch.optim.AdamW(layer.parameters(), lr=0.01, weight_decay=0.1, eps=epsilon)
    optimizer.step()
    assert isinstance(layer.weight, TrainingWeight)
    assert torch.isfinite(layer.weight.dequantize()).all()


def test_integer_compiler_key_tracks_code_and_metadata(monkeypatch):
    pytest.importorskip("torchao")
    from dev.benchmarks import _int8_weight

    weight = _int8_weight.TrainingWeight.from_float(torch.randn(8, 16))
    key = weight._stable_hash_for_caching()
    other_values = _int8_weight.TrainingWeight.from_float(torch.randn(8, 16))
    assert other_values._stable_hash_for_caching() == key
    assert weight.to(torch.float64)._stable_hash_for_caching() != key
    weight.requires_grad_(True)
    assert weight._stable_hash_for_caching() != key
    weight.requires_grad_(False)
    monkeypatch.setattr(_int8_weight, "_IMPLEMENTATION_HASH", "changed-backward-implementation")
    assert weight._stable_hash_for_caching() != key


def test_compiled_integer_parameter_gradients():
    import copy

    pytest.importorskip("torchao")
    if os.environ.get("RUN_CUDA_TESTS") != "1":
        pytest.skip("Set RUN_CUDA_TESTS=1 for compiled INT8 gradient validation")
    if not torch.cuda.is_available():
        pytest.fail("CUDA requested but unavailable")
    from dev.benchmarks.quantized_training import Layer

    torch.manual_seed(42)
    eager = torch.nn.Sequential(Layer(64, True, torch.float16), Layer(64, True, torch.float16))
    compiled = torch.compile(copy.deepcopy(eager), fullgraph=True)
    # Ordinary training input does not require gradients. A small mean-reduced
    # loss exercises scale products below FP16's representable range.
    inputs = torch.randn(32, 64, device="cuda", dtype=torch.float16)
    for model in (eager, compiled):
        (model(inputs).float().square().mean() / 1000).backward()
    for expected, actual in zip(eager.parameters(), compiled.parameters(), strict=True):
        assert actual.grad is not None
        assert torch.count_nonzero(actual.grad) > 0
        error = (actual.grad.float() - expected.grad.float()).norm() / expected.grad.float().norm()
        assert error < 0.04
