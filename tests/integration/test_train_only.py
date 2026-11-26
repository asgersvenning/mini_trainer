"""Opt-in partial training: ``train_only`` (frozen, eval-mode rest) and the detached learned head's separate clipping."""

import functools

import pytest
import torch

from mini_trainer.logging import configure_loggers
from mini_trainer.train import _train_only, main
from tests.integration.test_integration_train import MockBuilder, TinyMockModel

G = "fc.linear.parametrizations.weight.original0"
B = "fc.linear.bias"


class BatchNormMockModel(TinyMockModel):
    """TinyMockModel with BatchNorm in the trunk, so frozen running statistics are checked too."""

    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 4, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(4),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
        )


def run(tmp_path, name, epochs=1, model_kwargs=None, **kwargs):
    torch.manual_seed(0)
    main(
        input=str(tmp_path / "data"),
        output=str(tmp_path / "output"),
        epochs=epochs,
        device="cpu",
        dtype="float32",
        name=name,
        builder=MockBuilder,
        model_builder_kwargs={"model_type": BatchNormMockModel(), "pretrained": False, **(model_kwargs or {})},
        logger_builder_kwargs={"verbose": False, "logger_cls": configure_loggers()},
        ema=False,
        seed=42,
        **kwargs,
    )
    return torch.load(tmp_path / "output" / name / "weights" / "last.pt", weights_only=False)


@pytest.fixture
def base(tmp_path):
    for label in ("class_a", "class_b"):
        (tmp_path / "data" / label).mkdir(parents=True)
    run(tmp_path, "base", epochs=2)
    return tmp_path / "output" / "base" / "weights" / "last.pt"


def tensors(state):
    return {k: v for k, v in state.items() if isinstance(v, torch.Tensor)}


@pytest.mark.parametrize("rows", [False, True])
def test_train_only_moves_only_the_named_parameters(tmp_path, base, rows):
    """From a frozen-head checkpoint, train the learned head's g and b (and the rows): nothing else changes, BN included."""
    before = tensors(torch.load(base, weights_only=False))
    patterns = [G, B] + (["fc.linear.parametrizations.weight.original1"] if rows else [])
    after = tensors(
        run(
            tmp_path,
            f"only_{rows}",
            model_kwargs={"weights": str(base), "trainable_scale": True, "trainable_bias": True},
            train_only=patterns,
            head_summary_every=1,
        )
    )
    assert after.keys() == before.keys()
    changed = {k for k in before if not torch.equal(before[k], after[k])}
    assert changed == set(patterns)  # trunk, BN statistics and counters, hidden layer and (unless named) rows unchanged
    assert "features.1.running_mean" in before and "fc.hidden.weight" in before


def test_train_only_keeps_modules_without_trainable_parameters_in_eval_mode():
    model = BatchNormMockModel()
    model.fc = torch.nn.Sequential(torch.nn.Dropout(0.5), torch.nn.Linear(4, 2))
    _train_only(model, ["fc.1.bias"])
    model.train()
    assert not model.features.training and not model.features[1].training and not model.fc[0].training
    assert model.training and model.fc.training and model.fc[1].training
    assert [n for n, p in model.named_parameters() if p.requires_grad] == ["fc.1.bias"]
    with pytest.raises(ValueError, match="match no parameter"):
        _train_only(BatchNormMockModel(), ["fc.nothing"])


def test_train_only_excludes_freeze_backbone(tmp_path):
    with pytest.raises(ValueError, match="train_only"):
        main(input=str(tmp_path), train_only=[B], freeze_backbone=True)


def test_the_detached_head_does_not_change_the_main_training(tmp_path, monkeypatch):
    """With and without the second head, under clipping that always acts, the trunk and main head train identically."""
    import mini_trainer.trainer as trainer_module

    for label in ("class_a", "class_b"):
        (tmp_path / "data" / label).mkdir(parents=True)
    monkeypatch.setattr(trainer_module, "train_one_epoch", functools.partial(trainer_module.train_one_epoch, clip_grad_norm=1e-3))
    plain = tensors(run(tmp_path, "plain", epochs=2))
    second = tensors(run(tmp_path, "second", epochs=2, model_kwargs={"detached_learned_head": True}))
    assert set(second) - set(plain) == {"fc.detached_linear.scale", "fc.detached_linear.bias"}
    for k, v in plain.items():
        torch.testing.assert_close(second[k], v, rtol=0, atol=0, msg=k)
    assert not torch.equal(second["fc.detached_linear.scale"], torch.ones_like(second["fc.detached_linear.scale"]))
