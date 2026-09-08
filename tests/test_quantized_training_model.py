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
    nn.utils.parametrizations.weight_norm(model["norm"])
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


def test_training_entrypoint_checkpoint_and_inference(tmp_path):
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
        "seed": 42,
        "builder": DeterministicBuilder,
        "model_builder_kwargs": {"model_type": TinyMockModel(), "hidden": False, "droprate": 0, "normalized": False},
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
    args["name"] = "resumed"
    args["checkpoint"] = str(tmp_path / "quantized/weights/checkpoint_last.pth")
    args["model_builder_kwargs"]["model_type"] = TinyMockModel()
    main(**args)
    resumed = load_training_weights(tmp_path / "resumed/weights/checkpoint_last.pth")
    assert resumed["epoch"] == 1
    assert isinstance(resumed["model"]["fc.linear.weight"], TrainingWeight)


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
