import importlib

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from mini_trainer.modeling import Classifier, classification_module
from tests.test_integration_train import MockBuilder, TinyMockModel

train_module = importlib.import_module("mini_trainer.train")


class DeterministicBuilder(MockBuilder):
    @staticmethod
    def build_dataloader(batch_size, **kwargs):
        # Fixed ordering and no stochastic transforms isolate checkpoint state from RNG restoration.
        images = torch.linspace(-1, 1, 8 * 3 * 5 * 5).reshape(8, 3, 5, 5)
        labels = torch.arange(8) % 2
        dataset = TensorDataset(images, labels)
        return labels.numpy(), DataLoader(dataset, batch_size=batch_size), DataLoader(dataset, batch_size=batch_size)


def assert_state_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_state_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert type(actual) is type(expected)
        assert len(actual) == len(expected)
        for value, reference in zip(actual, expected):
            assert_state_equal(value, reference)
    else:
        assert actual == expected


@pytest.mark.parametrize(
    "ema",
    [
        False,
        pytest.param(
            True,
            marks=pytest.mark.xfail(
                strict=True,
                raises=RuntimeError,
                reason="EMA update after evaluation encounters differently shaped Classifier inference-cache buffers; see docs/roadmap.md",
            ),
        ),
    ],
)
def test_checkpoint_restore_and_continuation(tmp_path, monkeypatch, ema):
    torch.manual_seed(42)
    for label in ("class_a", "class_b"):
        (tmp_path / "data" / label).mkdir(parents=True)

    args = {
        "input": str(tmp_path / "data"),
        "output": str(tmp_path),
        "name": "uninterrupted",
        "epochs": 3,
        "device": "cpu",
        "dtype": "float32",
        "seed": 42,
        "builder": DeterministicBuilder,
        "model_builder_kwargs": {"model_type": TinyMockModel(), "hidden": False, "droprate": 0, "normalized": False},
        "dataloader_builder_kwargs": {"batch_size": 4},
        "lr_schedule_builder_kwargs": {"warmup_epochs": 0},
        "ema": ema,
        "ema_builder_kwargs": {"decay_rate": 0.8, "update_rate": 1, "distill_start": 100},
        "logger_builder_kwargs": {"verbose": False},
    }
    original_train = train_module.train
    captured = {}

    def capture_train(**kwargs):
        captured.update(kwargs)
        original_train(**kwargs)

    monkeypatch.setattr(train_module, "train", capture_train)
    train_module.main(**args)
    weights = tmp_path / "uninterrupted" / "weights"
    checkpoint = torch.load(weights / "checkpoint_0.pth", weights_only=True)
    expected = torch.load(weights / "checkpoint_last.pth", weights_only=True)
    assert checkpoint["epoch"] == 0
    assert all(state["state"] for state in checkpoint["optimizer"].values())
    assert checkpoint["lr_scheduler"]["last_epoch"] > 0
    if ema:
        assert checkpoint["model_ema"]["n_averaged"].item() > 0

    # Compare the live final model with reconstruction from the public weights artifact.
    live_model = captured["model_ema"].module if ema else captured["model"]
    live_model.eval()
    loaded, preprocess = Classifier.build(weights=str(weights / "last.pt"))
    loaded.eval()
    probe = torch.linspace(-0.75, 0.75, 3 * 3 * 5 * 5).reshape(3, 3, 5, 5)
    with torch.inference_mode():
        reference = live_model(captured["preprocess"](probe))
        torch.testing.assert_close(loaded(preprocess(probe)), reference, rtol=0, atol=0)
    assert classification_module(loaded).metadata["cls2idx"] == {"class_a": 0, "class_b": 1}

    restored = False

    def inspect_restore(**kwargs):
        nonlocal restored
        assert kwargs["start_epoch"] == 1
        for key in ("model", "optimizer", "lr_scheduler", "scaler"):
            assert_state_equal(kwargs[key].state_dict(), checkpoint[key])
        if ema:
            assert_state_equal(kwargs["model_ema"].state_dict(), checkpoint["model_ema"])
        restored = True
        original_train(**kwargs)

    monkeypatch.setattr(train_module, "train", inspect_restore)
    # The original total epoch budget is retained: changing it changes the LR schedule.
    args.update(name="resumed", checkpoint=str(weights / "checkpoint_0.pth"))
    args["model_builder_kwargs"] = {**args["model_builder_kwargs"], "model_type": TinyMockModel()}
    train_module.main(**args)
    assert restored
    actual = torch.load(tmp_path / "resumed" / "weights" / "checkpoint_last.pth", weights_only=True)
    assert_state_equal(actual, expected)


def test_ema_state_restored_before_continuation(tmp_path, monkeypatch):
    torch.manual_seed(42)
    for label in ("class_a", "class_b"):
        (tmp_path / "data" / label).mkdir(parents=True)
    args = {
        "input": str(tmp_path / "data"),
        "output": str(tmp_path),
        "name": "ema",
        "epochs": 1,
        "device": "cpu",
        "dtype": "float32",
        "seed": 42,
        "builder": DeterministicBuilder,
        "model_builder_kwargs": {"model_type": TinyMockModel(), "hidden": False, "droprate": 0, "normalized": False},
        "dataloader_builder_kwargs": {"batch_size": 4},
        "lr_schedule_builder_kwargs": {"warmup_epochs": 0},
        "ema": True,
        "ema_builder_kwargs": {"decay_rate": 0.8, "update_rate": 1, "distill_start": 100},
        "logger_builder_kwargs": {"verbose": False},
    }
    train_module.main(**args)
    checkpoint_path = tmp_path / "ema" / "weights" / "checkpoint_last.pth"
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    assert checkpoint["model_ema"]["n_averaged"].item() == 2
    restored = False

    def inspect_restore(**kwargs):
        nonlocal restored
        assert kwargs["start_epoch"] == 1
        for key in ("model", "model_ema", "optimizer", "lr_scheduler", "scaler"):
            assert_state_equal(kwargs[key].state_dict(), checkpoint[key])
        restored = True

    # Stop at the real continuation boundary to isolate restoration from the known EMA update bug.
    monkeypatch.setattr(train_module, "train", inspect_restore)
    args["checkpoint"] = str(checkpoint_path)
    args["epochs"] = 2
    args["model_builder_kwargs"] = {**args["model_builder_kwargs"], "model_type": TinyMockModel()}
    train_module.main(**args)
    assert restored
