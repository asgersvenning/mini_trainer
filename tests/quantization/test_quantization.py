"""Real W8A8 training, continuation and native integer inference regressions."""

import copy
import importlib.util
import json
from unittest.mock import Mock

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from mini_trainer.hierarchical.model import ConditionalClassifier, HierarchicalClassifier, IndependentClassifier
from mini_trainer.modeling import Classifier
from mini_trainer.modeling.quantization import load_int8, prepare_int8
from mini_trainer.trainer import train_one_epoch
from tests.training.test_checkpoint_contract import assert_state_equal

pytestmark = pytest.mark.skipif(importlib.util.find_spec("torchao") is None, reason="Install mini_trainer[quantization]")


def model(head=Classifier, normalized=True):
    kwargs = {}
    if head != Classifier:
        kwargs["sparse_masks"] = [torch.tensor([0, 0, 1, 1])]
    return head(in_features=8, out_features=4, hidden=False, normalized=normalized, **kwargs)


@pytest.mark.parametrize("qat", [False, True])
@pytest.mark.parametrize("head", [Classifier, HierarchicalClassifier, ConditionalClassifier, IndependentClassifier])
def test_heads_quantize_functional_parametrized_and_masked_linears(tmp_path, qat, head):
    torch.manual_seed(42)
    original = model(head)
    original.set_active_features([0, 2, 3])
    original.train()
    x = torch.randn(4, 8)
    before = copy.deepcopy(original.state_dict())
    prepared = prepare_int8(original, x, qat=qat)
    assert original.training
    assert_state_equal(original.state_dict(), before)
    output = prepared(x)
    if qat:
        sum(t.square().mean() for t in (output if isinstance(output, list) else [output])).backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in prepared.parameters())
    converted = prepared.convert()
    lowered, coverage = converted.lower(x)
    assert coverage["integer_kernels"]
    assert any(t.dtype == torch.int8 for t in converted.graph.buffers())
    with torch.no_grad(), torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        expected = lowered(x)
    assert "onednn::qlinear_pointwise" in {e.key for e in profile.key_averages()}
    path = converted.save(tmp_path / "int8", x, preprocessing={"input": "embeddings"}, calibration={"split": "train", "seed": 42})
    assert torch.export.load(path / "model.pt2").example_inputs is None
    reloaded, _ = load_int8(path).lower(x)
    with torch.no_grad():
        torch.testing.assert_close(reloaded(x), expected, rtol=0, atol=0)
    manifest = json.loads((path / "manifest.json").read_text())
    assert manifest["recipe"]["weight_bits"] == manifest["recipe"]["activation_bits"] == 8
    with pytest.raises(FileExistsError):
        converted.save(path, x, preprocessing={}, calibration={})


@pytest.mark.parametrize("normalized", [False, True])
def test_qat_resume_and_evaluation_do_not_recalibrate(tmp_path, normalized):
    torch.manual_seed(42)
    original = model(normalized=normalized)
    x = torch.randn(4, 8)
    labels = torch.tensor([0, 1, 2, 3])
    prepared = prepare_int8(original, x, qat=True)
    optimizer = torch.optim.AdamW(prepared.parameters(), lr=0.01)

    def step(p, opt):
        p.train()
        opt.zero_grad()
        torch.nn.functional.cross_entropy(p(x), labels).backward()
        opt.step()

    step(prepared, optimizer)
    prepared.eval()
    before = copy.deepcopy(prepared.state_dict())
    with torch.no_grad():
        prepared(x * 1000)  # Held-out outliers must not influence ranges or BN.
    assert_state_equal(prepared.state_dict(), before)
    prepared.train()
    prepared.freeze_observers()
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"model": prepared.state_dict(), "optimizer": optimizer.state_dict()}, checkpoint)
    step(prepared, optimizer)
    restored = prepare_int8(original, x, qat=True)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.01)
    state = torch.load(checkpoint, weights_only=True)
    restored.load_state_dict(state["model"])
    restored_optimizer.load_state_dict(state["optimizer"])
    step(restored, restored_optimizer)
    assert_state_equal(restored.state_dict(), prepared.state_dict())
    assert_state_equal(restored_optimizer.state_dict(), optimizer.state_dict())
    expected, _ = prepared.convert().lower(x)
    actual, _ = restored.convert().lower(x)
    with torch.no_grad():
        torch.testing.assert_close(actual(x), expected(x), rtol=0, atol=0)


def test_synthetic_qat_uses_training_loop_and_integer_inference():
    torch.manual_seed(42)

    # Two independent factors give an exact oracle; disjoint nuisance samples.
    def samples(seed):
        generator = torch.Generator().manual_seed(seed)
        labels = torch.arange(4).repeat(8)
        values = torch.randn(32, 8, generator=generator) * 0.02
        values[:, 0] += ((labels // 2) * 2 - 1) * 2
        values[:, 1] += ((labels % 2) * 2 - 1) * 2
        return values.reshape(32, 8, 1, 1), labels

    images, labels = samples(1)
    heldout, expected = samples(2)
    prepared = prepare_int8(torch.nn.Sequential(torch.nn.Flatten(), model(normalized=False)), images[:8], qat=True)
    optimizer = torch.optim.SGD(prepared.parameters(), lr=0.2, momentum=0.9)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1000)
    teacher = Mock()
    teacher.teach.return_value = torch.tensor(0.0)
    logger = Mock()
    logger.status.return_value = "QAT oracle"
    loader = DataLoader(TensorDataset(images, labels), batch_size=8)
    for epoch in range(8):
        train_one_epoch(prepared, teacher, torch.nn.CrossEntropyLoss(), optimizer, scaler, scheduler, loader, epoch, logger)
    assert scheduler.last_epoch == 32
    inference, coverage = prepared.convert().lower(heldout[:8])
    with torch.no_grad():
        predictions = torch.cat([inference(batch).argmax(1) for batch in heldout.split(8)])
    assert torch.equal(predictions, expected)
    assert coverage["integer_kernels"]["onednn.qlinear_pointwise.default"] == 1


def test_refuse_uncalibrated_and_empty_quantization():
    x = torch.randn(4, 8)
    prepared = prepare_int8(model(), x)
    with pytest.raises(ValueError, match="observe finite"):
        prepared.convert()
    with pytest.raises(ValueError, match="missing"):
        prepare_int8(torch.nn.Identity(), x)
    with pytest.raises(ValueError, match="CPU float32"):
        prepare_int8(model(), x.double())


@pytest.mark.parametrize("qat", [False, True])
def test_convolution_batchnorm_hidden_head_and_artifact_integrity(tmp_path, qat):
    torch.manual_seed(42)
    original = torch.nn.Sequential(
        torch.nn.Conv2d(3, 8, 3, padding=1),
        torch.nn.BatchNorm2d(8),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d(2),
        torch.nn.Flatten(),
        Classifier(32, 4, hidden=6, droprate=0.1, normalized=True),
    )
    x = torch.randn(4, 3, 8, 8)
    prepared = prepare_int8(original, x, qat=qat)
    if qat:
        prepared(x).square().mean().backward()
    else:
        with torch.no_grad():
            prepared(x)
    converted = prepared.convert()
    _, coverage = converted.lower(x)
    assert any("qconv_pointwise" in op for op in coverage["integer_kernels"])
    assert sum(count for op, count in coverage["integer_kernels"].items() if "qlinear" in op) == 2
    path = converted.save(tmp_path / "bundle", x, preprocessing={}, calibration={"split": "train"})
    (path / "model.pt2").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="checksum"):
        load_int8(path)


def test_reduced_range_recipe_and_full_range_resume_compatibility():
    torch.manual_seed(42)
    original = model()
    x = torch.randn(4, 8)
    portable = prepare_int8(original, x, qat=True)
    assert portable.recipe["activation_quant_max"] == 127
    activation_observers = [m for m in portable.graph.modules() if getattr(m, "quant_min", None) == 0]
    assert activation_observers
    assert all(m.quant_max == 127 for m in activation_observers)
    full = prepare_int8(original, x, qat=True, reduce_range=False)
    # Older recipes have no activation range fields; explicit full range retains that format.
    assert "activation_quant_max" not in full.recipe
    full_observers = [m for m in full.graph.modules() if getattr(m, "quant_min", None) == 0]
    assert full_observers and all(m.quant_max == 255 for m in full_observers)
    restored = prepare_int8(original, x, qat=True, reduce_range=False)
    restored.load_state_dict(full.state_dict())
    assert_state_equal(restored.state_dict(), full.state_dict())
    with pytest.raises(ValueError, match="recipe differs"):
        portable.load_state_dict(full.state_dict())
