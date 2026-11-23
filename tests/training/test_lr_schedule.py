import importlib

import pytest
import torch

from mini_trainer.builders import BaseBuilder
from tests.integration.test_integration_train import TinyMockModel
from tests.training.test_checkpoint_contract import DeterministicBuilder, assert_state_equal

train_module = importlib.import_module("mini_trainer.train")


def rates(epochs, steps_per_epoch, pretrained_backbone=True, **kwargs):
    """Learning rates of a head and a backbone group at every step of the schedule."""
    head, backbone = torch.nn.Parameter(torch.zeros(1)), torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([{"params": [head], "name": "head"}, {"params": [backbone], "name": "backbone"}], lr=0.5)
    scheduler = BaseBuilder.build_lr_scheduler(
        optimizer, epochs=epochs, steps_per_epoch=steps_per_epoch, pretrained_backbone=pretrained_backbone, **kwargs
    )
    out = []
    for _ in range(epochs * steps_per_epoch):
        out.append([group["lr"] for group in optimizer.param_groups])
        optimizer.step()
        scheduler.step()
    return out, scheduler


@pytest.mark.parametrize("pretrained_backbone", [True, False])
@pytest.mark.parametrize("epochs, warmup_epochs, steps_per_epoch", [(10, 1, 7), (5, 0.5, 3), (4, 0, 5), (20, 2, 13)])
def test_wsd_with_a_zero_length_stable_phase_reproduces_the_cosine_schedule(epochs, warmup_epochs, steps_per_epoch, pretrained_backbone):
    cosine, _ = rates(epochs, steps_per_epoch, pretrained_backbone, warmup_epochs=warmup_epochs)
    # A decay as long as the time after warmup, or longer, starts when warmup ends.
    for decay_epochs in (epochs - warmup_epochs, epochs):
        wsd, scheduler = rates(
            epochs, steps_per_epoch, pretrained_backbone, warmup_epochs=warmup_epochs, schedule="wsd", decay_epochs=decay_epochs
        )
        assert wsd == cosine  # exact, not approximate
        assert scheduler.decay_start_step == round(warmup_epochs * steps_per_epoch)


def test_wsd_holds_the_peak_rate_and_decays_over_the_last_epochs():
    lrs, scheduler = rates(10, 4, warmup_epochs=1, schedule="wsd", decay_epochs=2, min_factor=1e-3)
    head = [h for h, _ in lrs]
    backbone = [b for _, b in lrs]
    assert scheduler.decay_start_step == 32
    assert head[0] < head[3] < 0.5 and backbone[:4] == [0.0] * 4  # warmup: head rises, pretrained backbone waits
    assert head[4:33] == [0.5] * 29 and backbone[4:33] == [0.5] * 29  # stable until the decay's first step
    assert all(a > b for a, b in zip(head[32:], head[33:]))  # then decreasing over the last 8 steps
    assert head[39] == backbone[39] > 0.5e-3


def test_extending_a_wsd_run_keeps_every_rate_before_its_decay():
    short, _ = rates(6, 5, warmup_epochs=1, schedule="wsd", decay_epochs=2)
    longer, _ = rates(12, 5, warmup_epochs=1, schedule="wsd", decay_epochs=2)
    assert longer[:20] == short[:20]  # the short run's decay starts at step 20
    assert longer[20:50] == [[0.5, 0.5]] * 30


@pytest.mark.parametrize("pretrained_backbone", [True, False])
@pytest.mark.parametrize("epochs, warmup_epochs, steps_per_epoch", [(5, 0.5, 4), (4, 0, 5), (10, 1, 7)])
def test_linear_schedule_changes_only_the_decay_shape(epochs, warmup_epochs, steps_per_epoch, pretrained_backbone):
    kwargs = dict(warmup_epochs=warmup_epochs, start_factor=0.01, min_factor=0.0)
    cosine, _ = rates(epochs, steps_per_epoch, pretrained_backbone, **kwargs)
    linear, _ = rates(epochs, steps_per_epoch, pretrained_backbone, schedule="linear", **kwargs)
    assert rates(epochs, steps_per_epoch, pretrained_backbone, schedule="cosine", **kwargs)[0] == cosine  # the default
    total, warmup = epochs * steps_per_epoch, round(warmup_epochs * steps_per_epoch)
    assert linear[:warmup] == cosine[:warmup]  # same warmup (pretrained backbone waits at 0)
    assert linear[warmup] == pytest.approx(cosine[warmup])  # same peak
    assert cosine[warmup] == pytest.approx([0.5, 0.5])
    for group in (0, 1):  # then a straight line to min_factor at the step after the last
        decay = [lr[group] for lr in linear[warmup:]]
        expected = [0.5 * (1 - i / (total - warmup)) for i in range(total - warmup)]
        assert decay == pytest.approx(expected, abs=1e-12)
    if total - warmup > 2:
        assert linear[warmup + 1][0] < cosine[warmup + 1][0]  # linear leaves the peak faster than cosine


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"schedule": "poly"}, "Unknown learning-rate schedule"),
        ({"schedule": "wsd"}, "needs decay_epochs"),
        ({"schedule": "wsd", "decay_epochs": 0}, "needs decay_epochs"),
        ({"decay_epochs": 2}, "applies to schedule='wsd' only"),
    ],
)
def test_schedule_options_are_validated(kwargs, message):
    with pytest.raises(ValueError, match=message):
        rates(4, 2, warmup_epochs=1, **kwargs)


def train_args(tmp_path, name, epochs, decay_epochs, **extra):
    for label in ("class_a", "class_b"):
        (tmp_path / "data" / label).mkdir(parents=True, exist_ok=True)
    return {
        "input": str(tmp_path / "data"),
        "output": str(tmp_path),
        "name": name,
        "epochs": epochs,
        "device": "cpu",
        "dtype": "float32",
        "seed": 42,
        "builder": DeterministicBuilder,
        "model_builder_kwargs": {"model_type": TinyMockModel(), "hidden": False, "droprate": 0, "normalized": False},
        "dataloader_builder_kwargs": {"batch_size": 4},  # 8 samples: 2 steps per epoch
        "lr_schedule_builder_kwargs": {"warmup_epochs": 1, "schedule": "wsd", "decay_epochs": decay_epochs},
        "logger_builder_kwargs": {"verbose": False},
        **extra,
    }


def final_state(tmp_path, name):
    return torch.load(tmp_path / name / "weights" / "checkpoint_last.pth", weights_only=True)


def test_a_cooldown_branched_from_a_stable_checkpoint_equals_the_uninterrupted_run(tmp_path):
    torch.manual_seed(42)
    train_module.main(**train_args(tmp_path, "uninterrupted", epochs=10, decay_epochs=4))
    torch.manual_seed(42)
    train_module.main(**train_args(tmp_path, "long", epochs=14, decay_epochs=4))
    stable = tmp_path / "long" / "weights" / "checkpoint_5.pth"  # step 12; long's decay starts at step 20
    torch.manual_seed(42)
    train_module.main(**train_args(tmp_path, "branch", epochs=10, decay_epochs=4, checkpoint=str(stable)))
    assert_state_equal(final_state(tmp_path, "branch"), final_state(tmp_path, "uninterrupted"))


def test_an_unchanged_wsd_run_resumes_inside_its_decay(tmp_path):
    torch.manual_seed(42)
    train_module.main(**train_args(tmp_path, "uninterrupted", epochs=8, decay_epochs=4))
    inside_decay = tmp_path / "uninterrupted" / "weights" / "checkpoint_5.pth"  # step 12; decay from step 8
    torch.manual_seed(42)
    train_module.main(**train_args(tmp_path, "resumed", epochs=8, decay_epochs=4, checkpoint=str(inside_decay)))
    assert_state_equal(final_state(tmp_path, "resumed"), final_state(tmp_path, "uninterrupted"))


def test_a_branch_whose_decay_would_already_have_started_is_refused(tmp_path):
    torch.manual_seed(42)
    train_module.main(**train_args(tmp_path, "long", epochs=14, decay_epochs=4))
    stable = tmp_path / "long" / "weights" / "checkpoint_5.pth"  # step 12 at the peak rate
    with pytest.raises(ValueError, match="changes the learning rate at the step restored"):
        # epochs 8: the new decay would start at step 8, so the rate would drop at once at step 12.
        train_module.main(**train_args(tmp_path, "branch", epochs=8, decay_epochs=4, checkpoint=str(stable)))
