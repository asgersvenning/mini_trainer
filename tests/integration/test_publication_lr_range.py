"""LR probes retain warmup, observe overflow and distinguish parameter families."""

import pytest
import torch

from mini_trainer.trainer import _optimizer_step
from publication.experiments.training_ablations.lr_range import ProbeScaler, RangeBuilder, lr_factor


def test_range_starts_after_warmup_and_hold_keeps_backbone_ratio():
    lower, upper, warmup = 1e-5, 1.0, 128
    assert all(lr_factor(s, warmup, lower, upper, backbone=True) == 0 for s in range(warmup))
    assert upper * lr_factor(warmup - 1, warmup, lower, upper) < 0.0003
    values = [upper * lr_factor(s, warmup, lower, upper) for s in range(warmup, 2 * warmup)]
    assert values[0] == pytest.approx(lower)
    assert values[-1] == pytest.approx(upper)
    assert all(a < b for a, b in zip(values, values[1:]))
    for step in range(warmup, 2 * warmup):
        head = upper * lr_factor(step, warmup, lower, upper, hold=True)
        backbone = upper / 3 * lr_factor(step, warmup, lower, upper, backbone=True, hold=True)
        assert head == pytest.approx(upper)
        assert backbone == pytest.approx(head / 3)


def test_probe_observes_unscaled_gradients_and_amp_skips(monkeypatch):
    parameter = torch.nn.Parameter(torch.tensor([2.0]))
    optimizer = torch.optim.SGD([{"params": [parameter], "lr": 0.1, "name": "backbone"}])
    monkeypatch.setattr(RangeBuilder, "gradient_groups", {"backbone": [parameter], "projection": [], "classifier": []})
    scaler = ProbeScaler("cpu", init_scale=16)
    scaler.scale(parameter.square().sum()).backward()
    scaler.unscale_(optimizer)
    assert RangeBuilder.record["gradient_norms_before_clipping"]["backbone"] == pytest.approx(4)
    assert _optimizer_step(optimizer, scaler)
    assert not RangeBuilder.record["amp_skipped"]
    assert RangeBuilder.record["backbone_active"]
    optimizer.zero_grad()
    scaler.scale(parameter.sum() * float("inf")).backward()
    scaler.unscale_(optimizer)
    before = parameter.detach().clone()
    assert not _optimizer_step(optimizer, scaler)
    assert RangeBuilder.record["amp_skipped"]
    assert RangeBuilder.record["gradient_norms_before_clipping"]["backbone"] is None
    torch.testing.assert_close(parameter, before)


def test_probe_builder_runs_production_train_reload(tmp_path, monkeypatch):
    from publication.experiments.training_ablations import training
    from tests.integration.test_publication_ablations import test_tiny_train_reload_evaluate

    monkeypatch.setattr(training, "StudyBuilder", RangeBuilder)
    monkeypatch.setattr(RangeBuilder, "lower", 0.0001)
    monkeypatch.setattr(RangeBuilder, "upper", 0.001)
    monkeypatch.setattr(RangeBuilder, "hold", False)
    test_tiny_train_reload_evaluate(tmp_path, monkeypatch, True, "muon", True, "emla")
    import json

    rows = [json.loads(line) for line in (tmp_path / "study/attempt/lr-curve.jsonl").read_text().splitlines()]
    assert rows and any(row["backbone_active"] for row in rows)
    assert all(row["group_lrs"] and "loss" in row for row in rows)
