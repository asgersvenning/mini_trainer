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


def test_disabled_ema_does_not_preprocess_an_unused_teacher_batch():
    from mini_trainer.modeling.ema import EMATeacher

    torch.manual_seed(119)
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(2, 2))
    reference = copy.deepcopy(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
    images = torch.tensor([[[[1.0, 2.0]]], [[[3.0, 4.0]]]])
    targets = torch.tensor([0, 1])
    loader = DataLoader(TensorDataset(images, targets), batch_size=1)
    teacher = EMATeacher(enable=False, total_steps=2)
    teacher.teach = Mock(side_effect=AssertionError("Disabled teacher must not be invoked"))
    preprocess = Mock(side_effect=lambda batch: batch / 4)
    logger = Mock()
    logger.status.return_value = "disabled teacher regression"
    criterion = torch.nn.CrossEntropyLoss()
    train_one_epoch(
        model,
        teacher,
        criterion,
        optimizer,
        torch.amp.GradScaler("cpu", enabled=False),
        torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1),
        loader,
        0,
        logger,
        preprocess=preprocess,
        clip_grad_norm=None,
    )
    for batch, target in loader:
        reference_optimizer.zero_grad()
        criterion(reference(batch / 4), target).backward()
        reference_optimizer.step()
    teacher.teach.assert_not_called()
    assert preprocess.call_count == len(loader)
    assert all(call.kwargs["distillation_loss"] == 0.0 for call in logger.consume.call_args_list)
    assert_state_equal(model.state_dict(), reference.state_dict())


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
@pytest.mark.parametrize("compiled", [False, True])
def test_epoch_only_advances_scheduler_and_ema_after_updates(kind, scaled, device, compiled):
    torch.manual_seed(42)
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(2, 2)).to(device)
    optimizer = make_optimizer(kind, model.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    if compiled:
        from mini_trainer.training.compilation import compile_optimizer

        compile_optimizer(optimizer, backend="eager" if device.type == "cpu" else None)
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
    assert teacher.teach.call_count == len(loader)
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


def test_compiled_optimizer_scheduler_does_not_recompile_every_step():
    from torch._dynamo.testing import CompileCounter

    from mini_trainer.training.compilation import compile_optimizer

    torch._dynamo.reset()
    parameter = torch.nn.Parameter(torch.ones(4, 4))
    optimizer = torch.optim.SGD([parameter], lr=0.1, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 0.93**step)
    counter = CompileCounter()
    compile_optimizer(optimizer, backend=counter)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    frames_after_warmup = None
    for step in range(12):
        optimizer.zero_grad()
        parameter.square().sum().backward()
        assert _optimizer_step(optimizer, scaler)
        scheduler.step()
        if step == 2:
            frames_after_warmup = counter.frame_count
    assert frames_after_warmup and counter.frame_count == frames_after_warmup
    state = optimizer.state_dict()
    assert isinstance(state["param_groups"][0]["lr"], float)
    assert isinstance(optimizer.param_groups[0]["lr"], torch.Tensor)
    optimizer.load_state_dict(state)
    assert isinstance(optimizer.param_groups[0]["lr"], torch.Tensor)
    fresh = torch.optim.SGD([torch.nn.Parameter(parameter.detach().clone())], lr=1.0, momentum=0.9)
    fresh.load_state_dict(state)
    assert isinstance(fresh.param_groups[0]["lr"], float)


def test_compiled_foreach_adamw_requires_capturable_before_mutation():
    from mini_trainer.training.compilation import compile_optimizer

    parameter = torch.nn.Parameter(torch.ones(4, 4))
    optimizer = torch.optim.AdamW([parameter], lr=0.01, foreach=True)
    before = optimizer.step
    with pytest.raises(ValueError, match="requires capturable=True"):
        compile_optimizer(optimizer, backend="eager")
    assert optimizer.step == before
    assert isinstance(optimizer.param_groups[0]["lr"], float)
    assert not optimizer.state


def _assert_compiled_optimizer_state(actual, expected, key=None):
    if key == "lr":
        # Compiled checkpoints deliberately serialize tensor rates as numbers.
        # Compare those values exactly, including explicitly selected float32.
        actual = actual.item() if isinstance(actual, torch.Tensor) else actual
        expected = expected.item() if isinstance(expected, torch.Tensor) else expected
        assert actual == expected
    elif isinstance(actual, torch.Tensor):
        # Dynamo moves Adam's scalar step counter to CUDA. Check its exact
        # numeric state; all non-counter tensor devices must remain unchanged.
        if key == "step" and actual.numel() == expected.numel() == 1:
            expected = expected.to(actual.device)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-6)
    elif isinstance(actual, dict):
        # Non-default rate precision is checkpoint reconstruction metadata;
        # eager state has no such marker. Roundtrip tests verify it separately.
        actual_keys = actual.keys() - {"_mini_trainer_lr_dtype"}
        assert actual_keys == expected.keys() - {"_mini_trainer_lr_dtype"}
        for name in actual_keys:
            _assert_compiled_optimizer_state(actual[name], expected[name], name)
    elif isinstance(actual, (list, tuple)):
        assert type(actual) is type(expected) and len(actual) == len(expected)
        for left, right in zip(actual, expected, strict=True):
            _assert_compiled_optimizer_state(left, right)
    else:
        assert actual == expected


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("cudagraphs", [False, True])
def test_cuda_compiled_optimizer_matches_eager_updates(kind, cudagraphs):
    from mini_trainer.training.compilation import compile_optimizer

    if os.environ.get("RUN_CUDA_TESTS") != "1":
        pytest.skip("Set RUN_CUDA_TESTS=1 to compare compiled CUDA updates")
    assert torch.cuda.is_available()
    torch.manual_seed(71)
    initial = torch.randn(8, 8, device="cuda")
    parameters = [torch.nn.Parameter(initial.clone()) for _ in range(2)]
    # Native fused tensor-lr kernels require float32; use the same explicitly
    # chosen rate and scheduler arithmetic for both reference and candidate.
    rate = torch.tensor(0.01, device="cuda", dtype=torch.float32) if cudagraphs and kind.startswith("fused") else 0.01
    optimizers = [
        make_optimizer(kind, [parameter], lr=rate.clone() if isinstance(rate, torch.Tensor) else rate) for parameter in parameters
    ]
    schedulers = [torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.93) for opt in optimizers]
    compile_optimizer(optimizers[1], cudagraphs=cudagraphs)
    scalers = [torch.amp.GradScaler("cuda", init_scale=8, growth_interval=100) for _ in optimizers]
    for step in range(8):
        if cudagraphs and step == 6:
            optimizers[1].load_state_dict(copy.deepcopy(optimizers[1].state_dict()))
        gradient = torch.randn_like(initial)
        decisions = []
        for parameter, optimizer, scheduler, scaler in zip(parameters, optimizers, schedulers, scalers, strict=True):
            optimizer.zero_grad()
            scaler.scale((parameter * gradient).sum()).backward()
            if step == 2:
                parameter.grad.fill_(float("inf"))
            updated = _optimizer_step(optimizer, scaler)
            decisions.append(updated)
            if updated:
                scheduler.step()
        assert decisions[0] == decisions[1] == (step != 2)
        torch.testing.assert_close(parameters[0], parameters[1], rtol=1e-5, atol=2e-6)
        _assert_compiled_optimizer_state(optimizers[0].state_dict(), optimizers[1].state_dict())


def test_optimizer_graph_validation_precedes_mutation():
    from mini_trainer.training.compilation import compile_optimizer

    parameter = torch.nn.Parameter(torch.ones(4, 4))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    original_step = optimizer.step
    with pytest.raises(ValueError, match="one CUDA device"):
        compile_optimizer(optimizer, cudagraphs=True)
    assert optimizer.step == original_step and not optimizer.state
    assert not optimizer._optimizer_state_dict_post_hooks
    assert optimizer.param_groups[0]["lr"] == 0.1


@pytest.mark.parametrize("kind", ["fused_sgd", "fused_adamw"])
def test_cuda_fused_graph_rates_are_not_silently_rounded(kind):
    from mini_trainer.training.compilation import compile_optimizer

    if os.environ.get("RUN_CUDA_TESTS") != "1":
        pytest.skip("Set RUN_CUDA_TESTS=1 for native fused CUDA graph rate validation")
    parameter = torch.nn.Parameter(torch.ones(4, 4, device="cuda"))
    optimizer = make_optimizer(kind, [parameter])
    original_step = optimizer.step
    with pytest.raises(ValueError, match="explicit float32 tensor learning rate"):
        compile_optimizer(optimizer, cudagraphs=True)
    assert optimizer.step == original_step and not optimizer.state
    assert optimizer.param_groups[0]["lr"] == 0.01
    assert not optimizer._optimizer_state_dict_post_hooks


@pytest.mark.parametrize("kind", ["fused_sgd", "fused_adamw"])
def test_cuda_graph_checkpoint_retains_explicit_rate_precision(kind):
    from mini_trainer.training.compilation import compile_optimizer

    if os.environ.get("RUN_CUDA_TESTS") != "1":
        pytest.skip("Set RUN_CUDA_TESTS=1 to check native fused graph checkpoint precision")
    parameter = torch.nn.Parameter(torch.ones(4, 4, device="cuda"))
    optimizer = make_optimizer(kind, [parameter], lr=torch.tensor(0.01, dtype=torch.float32, device="cuda"))
    compile_optimizer(optimizer, cudagraphs=True)
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    state = copy.deepcopy(optimizer.state_dict())
    assert isinstance(state["param_groups"][0]["lr"], float)
    assert state["param_groups"][0]["_mini_trainer_lr_dtype"] == "float32"
    # Match the entrypoint order: restore first, then compile the fresh optimizer.
    restored = make_optimizer(kind, [parameter])
    restored.load_state_dict(state)
    compile_optimizer(restored, cudagraphs=True)
    for _ in range(4):
        restored.step()
        assert restored.param_groups[0]["lr"].dtype == torch.float32
        assert restored.param_groups[0]["lr"].device == parameter.device
    assert restored.state_dict()["param_groups"][0]["lr"] == state["param_groups"][0]["lr"]


def test_compiled_checkpoint_preserves_nondefault_rate_dtype_on_cpu():
    from mini_trainer.training.compilation import compile_optimizer

    parameter = torch.nn.Parameter(torch.ones(4, 4))
    optimizer = torch.optim.SGD([parameter], lr=torch.tensor(0.01, dtype=torch.float32))
    compile_optimizer(optimizer, backend="eager")
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    state = copy.deepcopy(optimizer.state_dict())
    assert state["param_groups"][0]["_mini_trainer_lr_dtype"] == "float32"
    restored = torch.optim.SGD([parameter], lr=0.1)
    restored.load_state_dict(state)
    observed = []
    restored.register_step_pre_hook(lambda opt, args, kwargs: observed.append(opt.param_groups[0]["lr"].dtype))
    compile_optimizer(restored, backend="eager")
    for _ in range(3):
        restored.step()
        assert restored.param_groups[0]["lr"].dtype == torch.float32
    assert observed == [torch.float32] * 3
    assert restored.state_dict()["param_groups"][0]["lr"] == state["param_groups"][0]["lr"]


def test_invalid_saved_rate_dtype_fails_before_compilation():
    from mini_trainer.training.compilation import compile_optimizer

    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.ones(4))], lr=0.1)
    optimizer.param_groups[0]["_mini_trainer_lr_dtype"] = "invalid"
    original_step = optimizer.step
    with pytest.raises(ValueError, match="Invalid saved learning-rate dtype"):
        compile_optimizer(optimizer)
    assert optimizer.step == original_step and not optimizer.state
    assert not optimizer._optimizer_state_dict_post_hooks
