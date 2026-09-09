"""Model selection, serialization and CUDA execution contracts for real QT."""

import importlib.util
import os

import pytest
import torch
from torch import nn

from mini_trainer.modeling.quantized_training import load_training_weights, prepare_quantized_training, restore_quantized_training

pytestmark = pytest.mark.skipif(importlib.util.find_spec("torchao") is None, reason="Install mini_trainer[quantization]")


def test_selection_preserves_ties_and_reports_float_operations():
    from mini_trainer.modeling._quantized_training import TrainingWeight

    model = nn.ModuleDict({"a": nn.Linear(8, 8), "b": nn.Linear(8, 8), "conv": nn.Conv2d(3, 3, 1), "norm": nn.Linear(8, 2)})
    model["b"].weight = model["a"].weight
    nn.utils.parametrizations.weight_norm(model["norm"], dim=1)
    report = prepare_quantized_training(model)
    assert report["quantized_modules"] == ["a", "b"]
    assert set(report["skipped_modules"]) == {"conv", "norm"}
    assert model["a"].weight is model["b"].weight
    assert isinstance(model["a"].weight, TrainingWeight)
    assert report["quantized_weight_bytes"] < report["reference_weight_bytes"]
    assert "conv.weight" in report["floating_parameter_names"]
    state = model.state_dict()
    assert state["_quantized_training"]["quantized_modules"] == ["a", "b"]


def test_invalid_selection_is_not_partially_applied():
    model = nn.Sequential(nn.Linear(8, 8), nn.Conv2d(3, 3, 1))
    before = model[0].weight
    with pytest.raises(ValueError, match="Unsupported"):
        prepare_quantized_training(model, module_names=["0", "1"])
    assert model[0].weight is before
    model = nn.ModuleDict({"embedding": nn.Embedding(8, 8), "linear": nn.Linear(8, 8)})
    model["linear"].weight = model["embedding"].weight
    with pytest.raises(ValueError, match="shared"):
        prepare_quantized_training(model, module_names=["linear"])
    assert model["linear"].weight is model["embedding"].weight


def test_quantized_checkpoint_restores_storage_and_recipe(tmp_path):
    from mini_trainer.modeling._quantized_training import TrainingWeight

    original = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 3))
    prepare_quantized_training(original)
    path = tmp_path / "weights.pt"
    torch.save(original.state_dict(), path)
    safe_before = set(torch.serialization.get_safe_globals())
    state = load_training_weights(path)
    assert set(torch.serialization.get_safe_globals()) == safe_before
    restored = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 3))
    restore_quantized_training(restored, state)
    restored.load_state_dict(state)
    for left, right in zip(original.parameters(), restored.parameters(), strict=True):
        if isinstance(left, TrainingWeight):
            assert isinstance(right, TrainingWeight)
            assert torch.equal(left.int_data, right.int_data)
            assert torch.equal(left.scale, right.scale)
        else:
            torch.testing.assert_close(left, right, rtol=0, atol=0)
    state["_quantized_training"]["quantized_modules"] = []
    with pytest.raises(RuntimeError, match="recipe"):
        restored.load_state_dict(state)


def cuda():
    if os.environ.get("RUN_CUDA_TESTS") != "1":
        pytest.skip("Set RUN_CUDA_TESTS=1 for model QT execution")
    if not torch.cuda.is_available():
        pytest.fail("CUDA requested but unavailable")
    return torch.device("cuda:0")


@pytest.mark.parametrize("autocast_dtype", [torch.float16, torch.bfloat16])
def test_model_autocast_single_sample_and_gradients(autocast_dtype):
    model = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1)).to(cuda())
    prepare_quantized_training(model)
    inputs = torch.randn(1, 64, device=cuda())
    with torch.autocast("cuda", dtype=autocast_dtype):
        output = model(inputs)
    assert output.dtype == autocast_dtype and output.shape == (1, 1)
    output.float().square().sum().backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    assert all(parameter.grad.dtype == torch.float32 for parameter in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    optimizer.step()
    with torch.no_grad(), torch.autocast("cuda", dtype=autocast_dtype):
        assert torch.isfinite(model(inputs)).all()


def test_masked_classifier_quantized_training_and_eval_cache():
    from mini_trainer.modeling import Classifier
    from mini_trainer.modeling._quantized_training import TrainingWeight

    model = Classifier(64, 4, hidden=32, normalized=False).to(cuda()).eval()
    inputs = torch.randn(4, 64, device=cuda())
    model(inputs)  # Populate the old floating-point evaluation cache.
    prepare_quantized_training(model)
    with torch.no_grad():
        model(inputs)
    assert isinstance(model._linear_weight, TrainingWeight)
    model.set_active_features([0, 2, 3])
    model.train()
    model(inputs).square().mean().backward()
    assert model.linear.weight.grad is not None
    assert torch.count_nonzero(model.linear.weight.grad[1]) == 0
    assert torch.count_nonzero(model.linear.weight.grad[[0, 2, 3]]) > 0


def test_mixed_muon_adamw_updates_and_counter():
    from mini_trainer.training.muon import MuonAuxAdamW

    model = nn.Linear(8, 8)
    prepare_quantized_training(model)
    optimizer = MuonAuxAdamW([{"name": "mixed", "params": list(model.parameters())}], lr=0.01, weight_decay=0.1)
    before = model.weight.int_data.clone()
    for parameter in model.parameters():
        parameter.grad = torch.randn(parameter.shape)
    optimizer.step()
    assert optimizer._step_count == 1
    assert optimizer.muon is not None and optimizer.adamw is not None
    assert not torch.equal(before, model.weight.int_data)
    state = optimizer.state_dict()
    restored = MuonAuxAdamW([{"name": "mixed", "params": list(model.parameters())}], lr=0.01, weight_decay=0.1)
    restored.load_state_dict(state)
    # The existing counter is process-local, not checkpoint state. Its role is
    # successful-step detection; restoring must not change that contract.
    before_count = restored._step_count
    restored.step()
    assert restored._step_count == before_count + 1


@pytest.mark.parametrize("normalized", [False, True])
@pytest.mark.parametrize("compiled_optimizer", [False, True])
@pytest.mark.parametrize("compile_mode", [None, "default", "reduce-overhead"])
def test_training_entrypoint_checkpoint_and_inference(tmp_path, normalized, compiled_optimizer, compile_mode):
    from mini_trainer.modeling import Classifier
    from mini_trainer.modeling._quantized_training import TrainingWeight
    from mini_trainer.train import main
    from tests.test_checkpoint_contract import DeterministicBuilder
    from tests.test_integration_train import TinyMockModel

    device = cuda()
    for label in ("class_a", "class_b"):
        (tmp_path / "data" / label).mkdir(parents=True)
    args = {
        "input": str(tmp_path / "data"),
        "output": str(tmp_path),
        "name": "quantized",
        "epochs": 1,
        "device": device,
        "dtype": "float16",
        "quantized_training": True,
        "compile_optimizer": compiled_optimizer,
        "compile": compile_mode is not None,
        "compile_mode": compile_mode,
        "seed": 42,
        "builder": DeterministicBuilder,
        "model_builder_kwargs": {"model_type": TinyMockModel(), "hidden": False, "droprate": 0, "normalized": normalized},
        "dataloader_builder_kwargs": {"batch_size": 4},
        "lr_schedule_builder_kwargs": {"warmup_epochs": 0},
        "ema": False,
        "logger_builder_kwargs": {"verbose": False},
    }
    main(**args)
    path = tmp_path / "quantized/weights/last.pt"
    restored, preprocess = Classifier.build(weights=str(path), device=device)
    assert any(isinstance(parameter, TrainingWeight) for parameter in restored.parameters())
    images = torch.randn(1, 3, 5, 5, device=device)
    restored.eval()
    with torch.no_grad():
        assert torch.isfinite(restored(preprocess(images))).all()
    state = load_training_weights(tmp_path / "quantized/weights/checkpoint_last.pth")
    assert state["optimizer"] and state["scaler"]
    # Resume through the ordinary training entry point. This checks state
    # plumbing, not identical stochastic continuation with a changed schedule.
    args["epochs"] = 2
    args["compile_optimizer"] = False
    args["compile"] = False
    args["compile_mode"] = None
    args["name"] = "resumed"
    args["checkpoint"] = str(tmp_path / "quantized/weights/checkpoint_last.pth")
    args["model_builder_kwargs"]["model_type"] = TinyMockModel()
    main(**args)
    resumed = load_training_weights(tmp_path / "resumed/weights/checkpoint_last.pth")
    assert resumed["epoch"] == 1
    key = "fc.linear.parametrizations.weight.original1" if normalized else "fc.linear.weight"
    assert isinstance(resumed["model"][key], TrainingWeight)


def test_quantized_regularizer_retains_weight_gradients():
    from mini_trainer.training.loss import class_weight_distribution_regularization

    model = nn.Linear(8, 4)
    prepare_quantized_training(model)
    reference = model.weight.dequantize().detach().requires_grad_()
    expected = class_weight_distribution_regularization(reference, sparse=False)
    actual = class_weight_distribution_regularization(model.weight, sparse=False)
    expected.backward()
    actual.backward()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(model.weight.grad, reference.grad)


def test_quantized_detach_in_inference_mode_preserves_alias():
    model = nn.Linear(8, 4)
    prepare_quantized_training(model)
    with torch.inference_mode():
        detached = model.weight.detach()
        assert not detached.is_inference()
        assert detached.int_data.data_ptr() == model.weight.int_data.data_ptr()
    with torch.no_grad():
        model.weight.mul_(0.9)
    assert torch.equal(detached.scale, model.weight.scale)


def test_quantized_class_similarity_uses_represented_values():
    from mini_trainer.modeling.distance import _class_similarity

    model = nn.Linear(8, 4)
    prepare_quantized_training(model)
    with torch.inference_mode():
        actual = _class_similarity(model.weight)
        expected = _class_similarity(model.weight.dequantize())
    torch.testing.assert_close(actual, expected)


def test_restoration_preserves_skipped_operation_reasons():
    original = nn.ModuleDict({"conv": nn.Conv2d(3, 4, 1), "linear": nn.Linear(4, 2)})
    prepare_quantized_training(original)
    state = original.state_dict()
    restored = nn.ModuleDict({"conv": nn.Conv2d(3, 4, 1), "linear": nn.Linear(4, 2)})
    restore_quantized_training(restored, state)
    restored.load_state_dict(state)
    assert restored._quantized_training_recipe["skipped_modules"] == original._quantized_training_recipe["skipped_modules"]


@pytest.mark.parametrize("negative_scale", [False, True])
def test_normalized_direction_matches_float_jacobian_without_saved_float_weights(negative_scale):
    from mini_trainer.modeling._quantized_training import TrainingWeight

    torch.manual_seed(19)
    direction = nn.Parameter(TrainingWeight.from_float(torch.randn(5, 17)))
    if negative_scale:
        with torch.no_grad():
            direction.scale.neg_()
    magnitude = nn.Parameter(torch.randn(5, 1))
    with torch.no_grad():
        magnitude[0].zero_()
    reference_direction = direction.detach().dequantize().requires_grad_()
    reference_magnitude = magnitude.detach().clone().requires_grad_()
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(lambda tensor: saved.append(tensor) or tensor, lambda tensor: tensor):
        actual = torch._weight_norm(direction, magnitude, 0)
    assert isinstance(actual, TrainingWeight)
    assert actual.int_data.data_ptr() == direction.int_data.data_ptr()
    assert not any(tensor.shape == direction.shape and tensor.is_floating_point() for tensor in saved)
    expected = torch._weight_norm(reference_direction, reference_magnitude, 0)
    torch.testing.assert_close(actual.dequantize(), expected)
    gradient = torch.randn_like(expected)
    actual.backward(gradient)
    expected.backward(gradient)
    torch.testing.assert_close(direction.grad, reference_direction.grad)
    torch.testing.assert_close(magnitude.grad, reference_magnitude.grad)


def test_normalized_checkpoint_restores_direction_and_magnitude(tmp_path):
    from mini_trainer.modeling._quantized_training import TrainingWeight

    def make():
        return nn.utils.parametrizations.weight_norm(nn.Linear(17, 5))

    original = make()
    report = prepare_quantized_training(original)
    assert report["quantized_modules"] == [""]
    assert report["floating_parameter_names"] == ["bias", "parametrizations.weight.original0"]
    path = tmp_path / "normalized.pt"
    torch.save(original.state_dict(), path)
    state = load_training_weights(path)
    restored = make()
    restore_quantized_training(restored, state)
    restored.load_state_dict(state)
    assert isinstance(restored.parametrizations.weight.original1, TrainingWeight)
    torch.testing.assert_close(restored.weight.dequantize(), original.weight.dequantize(), rtol=0, atol=0)


def test_invalid_normalized_direction_does_not_partially_prepare():
    model = nn.Sequential(nn.Linear(8, 8), nn.utils.parametrizations.weight_norm(nn.Linear(8, 4)))
    original = model[0].weight
    with torch.no_grad():
        model[1].parametrizations.weight.original1[0].zero_()
    with pytest.raises(ValueError, match="nonzero directions"):
        prepare_quantized_training(model)
    assert model[0].weight is original


@pytest.mark.parametrize("compiled", [False, True])
def test_normalized_classifier_integer_training_and_masked_inference(compiled):
    from mini_trainer.modeling import Classifier
    from mini_trainer.modeling._quantized_training import TrainingWeight

    torch.manual_seed(23)
    model = Classifier(64, 4, hidden=False, normalized=True).to(cuda())
    report = prepare_quantized_training(model)
    assert report["quantized_modules"] == ["linear"]
    parameter = model.linear.parametrizations.weight.original1
    assert isinstance(parameter, TrainingWeight)
    inputs = torch.randn(8, 64, device=cuda())
    target = torch.arange(8, device=cuda()) % 4
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    forward = torch.compile(model, fullgraph=True) if compiled else model
    before = parameter.dequantize().detach().clone()
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = torch.nn.functional.cross_entropy(forward(inputs), target)
        loss.backward()
        assert torch.isfinite(parameter.grad).all() and parameter.grad.norm() > 0
        optimizer.step()
    assert not torch.equal(before, parameter.dequantize())
    model.eval()
    with torch.inference_mode():
        complete = model(inputs[:1])
        model.set_active_features([0, 2, 3])
        selected = model(inputs[:1])
        torch.testing.assert_close(selected, complete[:, [0, 2, 3]])


def test_tensor_scalar_decay_preserves_integer_codes_and_rng():
    from mini_trainer.modeling._quantized_training import TrainingWeight

    weight = nn.Parameter(TrainingWeight.from_float(torch.randn(5, 17)))
    codes = weight.int_data.clone()
    scales = weight.scale.clone()
    rng = torch.get_rng_state()
    with torch.no_grad():
        weight.mul_(torch.tensor(0.97, dtype=torch.float64))
    assert torch.equal(weight.int_data, codes)
    assert torch.equal(torch.get_rng_state(), rng)
    torch.testing.assert_close(weight.scale, scales * 0.97)


@pytest.mark.parametrize("operation", ["add", "addcdiv"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_cuda_storage_update_rounding_versions_and_rng(operation, dtype):
    from mini_trainer.modeling._quantized_training import TrainingWeight

    if os.environ.get("RUN_CUDA_TESTS") != "1":
        pytest.skip("Set RUN_CUDA_TESTS=1 to verify fused storage updates")
    assert torch.cuda.is_available()
    torch.manual_seed(29)
    weight = nn.Parameter(TrainingWeight.from_float(torch.randn(33, 67, device="cuda", dtype=dtype)))
    update = torch.randn_like(weight.dequantize())
    denominator = update.abs() + 1

    def apply(target):
        with torch.no_grad():
            if operation == "add":
                return target.add_(update, alpha=-0.03)
            return target.addcdiv_(update, denominator, value=-0.03)

    # Warm compilation before saving RNG; compilation itself is outside the
    # update's reproducibility contract.
    apply(nn.Parameter(weight.detach().clone()))
    represented = weight.dequantize().detach()
    expected = represented.add(update, alpha=-0.03) if operation == "add" else represented.addcdiv(update, denominator, value=-0.03)
    replica = nn.Parameter(weight.detach().clone())
    before_version = weight._version
    before_codes_version = weight.int_data._version
    rng = torch.cuda.get_rng_state()
    assert apply(weight) is weight
    torch.cuda.set_rng_state(rng)
    apply(replica)
    assert weight._version > before_version
    assert weight.int_data._version > before_codes_version
    assert torch.equal(weight.int_data, replica.int_data)
    assert torch.equal(weight.scale, replica.scale)
    # Stochastic rounding differs by at most one code interval, plus the
    # floating arithmetic's precision. There is no floating master parameter.
    bound = expected.abs().amax(1, keepdim=True) / 127 + 4 * torch.finfo(dtype).eps
    assert torch.all((weight.dequantize() - expected).abs() <= bound)
    assert weight.int_data.dtype == torch.int8
    assert weight.scale.shape == (33,)


def test_cuda_storage_update_invalidates_saved_weight():
    from mini_trainer.modeling._quantized_training import TrainingWeight

    if os.environ.get("RUN_CUDA_TESTS") != "1":
        pytest.skip("Set RUN_CUDA_TESTS=1 to verify saved-tensor invalidation")
    assert torch.cuda.is_available()
    weight = nn.Parameter(TrainingWeight.from_float(torch.randn(16, 32, device="cuda")))
    inputs = torch.randn(8, 32, device="cuda", requires_grad=True)
    output = nn.functional.linear(inputs, weight)
    with torch.no_grad():
        weight.add_(torch.ones_like(weight.dequantize()), alpha=-0.01)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        output.sum().backward()


def test_cuda_storage_kernel_reused_across_parameter_objects_and_rates(monkeypatch):
    from torch._dynamo.testing import CompileCounterWithBackend

    from mini_trainer.modeling import _quantized_training as backend

    if os.environ.get("RUN_CUDA_TESTS") != "1":
        pytest.skip("Set RUN_CUDA_TESTS=1 to verify storage-kernel reuse")
    assert torch.cuda.is_available()
    counter = CompileCounterWithBackend("inductor")
    operation = backend.update_int8_rows_

    def update_rows(codes, scales, update, alpha, denominator):
        operation(codes, scales, update, alpha, denominator)

    kernel = torch.compile(update_rows, backend=counter, fullgraph=True, dynamic=True)
    monkeypatch.setattr(backend, "update_int8_rows_", kernel)
    weights = [nn.Parameter(backend.TrainingWeight.from_float(torch.randn(33, 67, device="cuda"))) for _ in range(12)]
    optimizer = torch.optim.SGD([{"params": [weight], "lr": 0.01 / (index + 1)} for index, weight in enumerate(weights)], momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.93)
    after_warmup = None
    for step in range(4):
        for weight in weights:
            weight.grad = torch.ones(weight.shape, device="cuda")
        optimizer.step()
        scheduler.step()
        if step == 0:
            after_warmup = counter.frame_count
    assert after_warmup and counter.frame_count == after_warmup


def test_cuda_local_matmul_tuning_bounds_temporary_memory(monkeypatch):
    from torchao.prototype.quantized_training.int8_mm import _scaled_int8_mm_kernel as upstream

    from mini_trainer.modeling._quantized_matmul import _kernel
    from mini_trainer.modeling._quantized_training import scaled_int8_mm

    if os.environ.get("RUN_CUDA_TESTS") != "1":
        pytest.skip("Set RUN_CUDA_TESTS=1 to verify first-use tuning memory")
    assert torch.cuda.is_available()
    assert _kernel is not upstream and _kernel.fn is upstream.fn
    # Force actual tuning rather than accepting an earlier process's disk cache.
    monkeypatch.setattr(_kernel, "cache", {})
    monkeypatch.setattr(_kernel, "cache_results", False)
    left = torch.randint(-127, 128, (17, 129), dtype=torch.int8, device="cuda")
    right = torch.randint(-127, 128, (19, 129), dtype=torch.int8, device="cuda").T
    rows = torch.rand(17, device="cuda")
    columns = torch.rand(19, device="cuda")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    result = scaled_int8_mm(left, right, rows, columns)
    torch.cuda.synchronize()
    extra_peak = torch.cuda.max_memory_allocated() - before
    # This tiny product must not trigger the old tuner's 256 MiB flush buffer.
    assert extra_peak < 16 * 1024**2
    product = left.cpu().to(torch.int64) @ right.cpu().to(torch.int64)
    expected = product.float().to("cuda") * rows[:, None] * columns[None, :]
    torch.testing.assert_close(result, expected)


@pytest.mark.parametrize("kind", ["sgd", "adamw"])
def test_cuda_compiled_optimizer_handles_many_quantized_groups(kind):
    from torch._dynamo.testing import CompileCounterWithBackend

    from mini_trainer.modeling._quantized_training import TrainingWeight
    from mini_trainer.training.compilation import compile_optimizer

    if os.environ.get("RUN_CUDA_TESTS") != "1":
        pytest.skip("Set RUN_CUDA_TESTS=1 to verify many-group optimizer compilation")
    assert torch.cuda.is_available()
    # Measure this optimizer's frames, independently of earlier tests' Dynamo
    # caches/skip decisions. Never reset between groups or measured updates.
    torch._dynamo.reset()
    weights = [nn.Parameter(TrainingWeight.from_float(torch.randn(64, 128, device="cuda"))) for _ in range(12)]
    groups = [{"params": [weight], "lr": 0.01 / (index + 1)} for index, weight in enumerate(weights)]
    optimizer = torch.optim.SGD(groups, momentum=0.9, weight_decay=0.1) if kind == "sgd" else torch.optim.AdamW(groups, foreach=False)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.93)
    counter = CompileCounterWithBackend("inductor")
    compile_optimizer(optimizer, backend=counter)
    after_warmup = None
    # Fail on fallback; some compiled frames alone would not establish that
    # every parameter group can execute without exhausting Dynamo's cache.
    with torch._dynamo.config.patch(fail_on_recompile_limit_hit=True):
        for step in range(6):
            for weight in weights:
                weight.grad = torch.ones(weight.shape, device="cuda")
            optimizer.step()
            scheduler.step()
            if step == 2:
                after_warmup = counter.frame_count
    assert after_warmup and counter.frame_count == after_warmup


@pytest.mark.parametrize("position", [0, 1, 2])
def test_fma_uses_represented_weights_without_mutation_or_rounding(position):
    from mini_trainer.modeling._quantized_training import TrainingWeight

    values = [torch.randn(4, 7) for _ in range(3)]
    weight = nn.Parameter(TrainingWeight.from_float(values[position]))
    values[position] = weight
    codes = weight.int_data.clone()
    scales = weight.scale.clone()
    version = weight._version
    rng = torch.get_rng_state()
    with torch.no_grad():
        expected = torch.ops.prims.fma(*(value.dequantize() if value is weight else value for value in values))
        actual = torch.ops.prims.fma(*values)
    assert not isinstance(actual, TrainingWeight)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(weight.int_data, codes) and torch.equal(weight.scale, scales)
    assert weight._version == version
    assert torch.equal(torch.get_rng_state(), rng)


@pytest.mark.parametrize("kind", ["sgd", "adamw"])
def test_cuda_compiled_quantized_update_matches_float_before_rounding(kind):
    from mini_trainer.modeling._quantized_training import TrainingWeight
    from mini_trainer.training.compilation import compile_optimizer

    if os.environ.get("RUN_CUDA_TESTS") != "1":
        pytest.skip("Set RUN_CUDA_TESTS=1 to verify compiled QT update arithmetic")
    assert torch.cuda.is_available()
    torch.manual_seed(107)
    weight = nn.Parameter(TrainingWeight.from_float(torch.randn(32, 64, device="cuda")))
    reference = nn.Parameter(weight.dequantize().detach().clone())
    cls = torch.optim.SGD if kind == "sgd" else torch.optim.AdamW
    options = {"lr": 0.03, "weight_decay": 0.1, "foreach": False}
    if kind == "sgd":
        options.update(momentum=0.9, nesterov=True)
    optimizer, eager = cls([weight], **options), cls([reference], **options)
    schedulers = [torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.93) for opt in (optimizer, eager)]
    compile_optimizer(optimizer)
    for _ in range(6):
        with torch.no_grad():
            reference.copy_(weight.dequantize())
        gradient = torch.randn_like(reference)
        weight.grad, reference.grad = gradient.clone(), gradient.clone()
        optimizer.step()
        eager.step()
        for scheduler in schedulers:
            scheduler.step()
        bound = reference.detach().abs().amax(1, keepdim=True) / 127 + 1e-6
        assert torch.all((weight.dequantize() - reference).abs() <= bound)
        for key, value in optimizer.state[weight].items():
            expected = eager.state[reference][key]
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(value, expected.to(value.device), rtol=1e-5, atol=2e-6)
            else:
                assert value == expected


def test_cuda_compiled_stochastic_rounding_preserves_sub_code_updates():
    from mini_trainer.modeling._quantized_training import TrainingWeight
    from mini_trainer.training.compilation import compile_optimizer

    if os.environ.get("RUN_CUDA_TESTS") != "1":
        pytest.skip("Set RUN_CUDA_TESTS=1 to verify sub-code compiled QT updates")
    assert torch.cuda.is_available()
    torch.manual_seed(109)
    weight = nn.Parameter(TrainingWeight.from_float(torch.ones(128, 1024, device="cuda")))
    optimizer = torch.optim.SGD([weight], lr=1, foreach=False)
    compile_optimizer(optimizer)
    gradient = torch.full(weight.shape, -0.25 / 127, device="cuda")
    gradient[:, 0] = 0
    previous_codes = None
    for _ in range(4):
        # Hold each row's maximum fixed, then request a quarter-code increase
        # elsewhere. Rounding must retain that signal statistically rather than
        # deterministically dropping it or biasing it upward.
        with torch.no_grad():
            weight.int_data.zero_()
            weight.int_data[:, 0] = 127
            weight.scale.fill_(1 / 127)
        weight.grad = gradient
        rng = torch.cuda.get_rng_state()
        optimizer.step()
        codes = weight.int_data[:, 1:]
        assert torch.all((codes == 0) | (codes == 1))
        assert abs(codes.float().mean().item() - 0.25) < 0.01
        assert not torch.equal(rng, torch.cuda.get_rng_state())
        if previous_codes is not None:
            assert not torch.equal(codes, previous_codes)
        previous_codes = codes.clone()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("compiled", [False, True])
def test_cuda_functional_requantization_preserves_independent_rounding(dtype, compiled):
    if os.environ.get("RUN_CUDA_TESTS") != "1":
        pytest.skip("Set RUN_CUDA_TESTS=1 to verify functional CUDA requantization")
    from mini_trainer.modeling._quantized_update import quantize_int8_rows

    values = torch.full((128, 2048), 0.25, device="cuda", dtype=dtype)[:, ::2]
    values[:, 0] = 127

    def twice(values):
        return quantize_int8_rows(values), quantize_int8_rows(values)

    run = torch.compile(twice, fullgraph=True) if compiled else twice
    run(values)  # Initialize compiler state before checking generator replay.
    rng = torch.cuda.get_rng_state()
    first, second = run(values)
    after = torch.cuda.get_rng_state()
    assert not torch.equal(rng, after)
    assert not torch.equal(first[0], second[0])
    for codes, scales in (first, second):
        assert codes.dtype == torch.int8 and scales.dtype == dtype
        assert torch.all(scales == 1) and torch.all(codes[:, 0] == 127)
        assert torch.all((codes[:, 1:] == 0) | (codes[:, 1:] == 1))
        assert abs(codes[:, 1:].float().mean().item() - 0.25) < 0.01
    torch.cuda.set_rng_state(rng)
    replay = run(values)
    for expected, actual in zip((first, second), replay, strict=True):
        for left, right in zip(expected, actual, strict=True):
            torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_cuda_weight_copy_preserves_storage_aliases_and_source():
    if os.environ.get("RUN_CUDA_TESTS") != "1":
        pytest.skip("Set RUN_CUDA_TESTS=1 to verify CUDA weight copy semantics")
    from mini_trainer.modeling._quantized_training import TrainingWeight

    codes = torch.zeros((4, 14), device="cuda", dtype=torch.int8)[:, ::2]
    scales = torch.ones(8, device="cuda")[::2]
    weight = nn.Parameter(TrainingWeight(codes, scales))
    alias = weight.detach()
    values = torch.randn((4, 14), device="cuda")[:, ::2]
    before = values.clone()
    version = weight._version
    with torch.no_grad():
        assert weight.copy_(values) is weight
    assert weight._version > version and alias._version == weight._version
    assert weight.int_data.data_ptr() == codes.data_ptr()
    assert weight.scale.data_ptr() == scales.data_ptr()
    torch.testing.assert_close(values, before, rtol=0, atol=0)
    torch.testing.assert_close(alias.dequantize(), weight.dequantize(), rtol=0, atol=0)
    bound = values.abs().amax(1, keepdim=True) / 127 + 1e-6
    assert torch.all((weight.dequantize() - values).abs() <= bound)


@pytest.mark.parametrize("mode", ["default", "reduce-overhead"])
def test_normalized_compilation_preserves_embedding_loss_gradients(mode):
    from mini_trainer.modeling import Classifier, EmbeddingContext

    torch.manual_seed(19)
    model = Classifier(64, 4, hidden=False, normalized=True).to(cuda())
    prepare_quantized_training(model)
    forward = torch.compile(model, mode=mode, fullgraph=True)
    data = torch.randn(8, 64, device=cuda())
    target = torch.arange(8, device=cuda()) % 4
    results = []
    for call in (model, forward):
        model.zero_grad(set_to_none=True)
        inputs = data.clone().requires_grad_()
        # Isolate graph-boundary gradients from AMP fusion rounding, which can
        # move normalized activations across an INT8 quantization threshold.
        with EmbeddingContext():
            scores = call(inputs)
            embeddings = EmbeddingContext.get()
            assert embeddings is not None and embeddings.requires_grad
            loss = torch.nn.functional.cross_entropy(scores, target) + embeddings[:, 0].sum() * 0.1
            loss.backward()
        results.append(
            (scores.detach().clone(), inputs.grad.clone(), [None if p.grad is None else p.grad.clone() for p in model.parameters()])
        )
    assert model.linear.parametrizations.weight.original1.grad.norm() > 0
    torch.testing.assert_close(results[0], results[1], rtol=1e-3, atol=1e-4)
