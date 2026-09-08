"""Optimizer, scheduler and EMA gating with real GradScaler overflow/scale growth."""

import copy
import os
from unittest.mock import Mock

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from mini_trainer.trainer import _optimizer_step, train_one_epoch
from mini_trainer.training import MuonAuxAdamW
from tests.test_checkpoint_contract import assert_state_equal

KINDS = ["muon", "adamw", "sgd", "fused_adamw", "fused_sgd"]


def make_optimizer(kind, parameters, lr=0.01):
    if kind == "muon":
        return MuonAuxAdamW([{"params": list(parameters), "name": "head"}], lr=lr, weight_decay=0.01)
    if "adamw" in kind:
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=0.01, fused=kind.startswith("fused"))
    return torch.optim.SGD(parameters, lr=lr, momentum=0.9, weight_decay=0.01, fused=kind.startswith("fused"))


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda":
        if os.environ.get("RUN_CUDA_TESTS") != "1":
            pytest.skip("Set RUN_CUDA_TESTS=1 and expose CUDA to exercise hardware AMP regressions")
        if not torch.cuda.is_available():
            pytest.fail("CUDA tests requested but no accessible CUDA device is available")
    return torch.device(request.param)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("scaled", [False, True])
def test_epoch_only_advances_scheduler_and_ema_after_updates(kind, scaled, device):
    torch.manual_seed(42)
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(2, 2)).to(device)
    optimizer = make_optimizer(kind, model.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    scaler = torch.amp.GradScaler(device.type, enabled=scaled, init_scale=8, growth_interval=1)
    images = torch.tensor([[[[1.0, 2.0]]]]).repeat(3, 1, 1, 1)
    loader = DataLoader(TensorDataset(images, torch.tensor([0, 1, 0])), batch_size=1)
    grad_calls = 0

    def overflow_once(grad):
        nonlocal grad_calls
        grad_calls += 1
        return torch.full_like(grad, float("inf")) if scaled and grad_calls == 2 else grad

    gradient_hook = model[1].weight.register_hook(overflow_once)
    external_calls = []
    external_hook = optimizer.register_step_post_hook(lambda *args: external_calls.append(True))
    initial_hooks = len(optimizer._optimizer_step_post_hooks)
    snapshots = []
    logger = Mock()
    logger.status.return_value = "step regression"
    logger.consume.side_effect = lambda **kwargs: snapshots.append(
        (copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict()), scheduler.last_epoch, scaler.get_scale())
    )
    teacher = Mock()
    teacher.teach.return_value = torch.tensor(0.0, device=device)
    precision = (torch.float16 if device.type == "cuda" else torch.bfloat16) if scaled else torch.float32
    train_one_epoch(
        model,
        teacher,
        torch.nn.CrossEntropyLoss(),
        optimizer,
        scaler,
        scheduler,
        loader,
        epoch=2,
        logger=logger,
        device=device,
        dtype=precision,
    )
    gradient_hook.remove()
    assert len(optimizer._optimizer_step_post_hooks) == initial_hooks
    external_hook.remove()
    expected_steps = [6, 8] if scaled else [6, 7, 8]
    assert [call.args[0] for call in teacher.update_parameters.call_args_list] == expected_steps
    assert all(call.args[1] is model for call in teacher.update_parameters.call_args_list)
    assert scheduler.last_epoch == len(expected_steps)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.01 * 0.5 ** len(expected_steps))
    if scaled:
        assert [snapshot[3] for snapshot in snapshots] == [16, 8, 16]
        assert_state_equal(snapshots[1][0], snapshots[0][0])
        assert_state_equal(snapshots[1][1], snapshots[0][1])
        assert snapshots[1][2] == snapshots[0][2] == 1
        assert any(not torch.equal(snapshots[2][0][key], snapshots[1][0][key]) for key in snapshots[2][0])
    # Fused optimizers call step even on overflow, but the kernel must not update.
    assert len(external_calls) == (3 if kind.startswith("fused") or not scaled else 2)
    if kind == "muon":
        assert optimizer._step_count == len(expected_steps)


@pytest.mark.parametrize("kind", KINDS)
def test_zero_lr_still_counts_as_a_completed_step(kind):
    parameter = torch.nn.Parameter(torch.ones(2, 2))
    optimizer = make_optimizer(kind, [parameter], lr=0.0)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    parameter.sum().backward()
    assert _optimizer_step(optimizer, scaler)
    torch.testing.assert_close(parameter, torch.ones_like(parameter), rtol=0, atol=0)


def test_step_failure_removes_hook_and_does_not_update_scaler():
    class BrokenSGD(torch.optim.SGD):
        def step(self, closure=None):
            raise RuntimeError("optimizer failed")

    optimizer = BrokenSGD([torch.nn.Parameter(torch.ones(1))], lr=0.1)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    scaler.update = Mock(wraps=scaler.update)
    with pytest.raises(RuntimeError, match="optimizer failed"):
        _optimizer_step(optimizer, scaler)
    assert not optimizer._optimizer_step_post_hooks
    scaler.update.assert_not_called()


def test_legacy_custom_amp_contract_rejected_before_step():
    class LegacySGD(torch.optim.SGD):
        _step_supports_amp_scaling = True

        def step(self, closure=None, grad_scaler=None):
            pytest.fail("Legacy AMP step must not run without reliable skip detection")

    optimizer = LegacySGD([torch.nn.Parameter(torch.ones(1))], lr=0.1)
    with pytest.raises(NotImplementedError, match="Legacy AMP"):
        _optimizer_step(optimizer, torch.amp.GradScaler("cpu"))
    assert not optimizer._optimizer_step_post_hooks


@pytest.mark.parametrize("kind", KINDS)
def test_overflow_detected_even_when_scale_cannot_back_off_further(kind):
    parameter = torch.nn.Parameter(torch.ones(2, 2))
    optimizer = make_optimizer(kind, [parameter])
    # Prolonged overflow can underflow GradScaler's scale to zero. Comparing
    # old/new scales would then incorrectly treat a skipped update as successful.
    scaler = torch.amp.GradScaler("cpu", init_scale=0.0)
    scaler.scale(parameter.sum()).backward()
    parameter.grad.fill_(float("inf"))
    scaler.unscale_(optimizer)
    assert not _optimizer_step(optimizer, scaler)
    assert scaler.get_scale() == 0.0
    torch.testing.assert_close(parameter, torch.ones_like(parameter), rtol=0, atol=0)
    assert not optimizer._optimizer_step_post_hooks
