import pytest
import torch

from mini_trainer.training.muon import Muon, _adjust_lr, _to_scalar, _zeropower_via_newtonschulz


def test_to_scalar():
    assert _to_scalar(0.5) == 0.5
    assert _to_scalar(torch.tensor(0.5)) == 0.5
    assert _to_scalar(torch.tensor([0.5])) == 0.5
    values = torch.tensor([0.25, 0.5])
    torch.testing.assert_close(_to_scalar(values), values)


def test_zeropower_via_newtonschulz():
    result = _zeropower_via_newtonschulz(torch.eye(4), ns_coefficients=(3.4445, -4.7750, 2.0315), ns_steps=5, eps=1e-7)
    assert result.shape == (4, 4)
    assert torch.isfinite(result).all()


@pytest.mark.parametrize(
    "mode,shape,expected",
    [("original", [10, 10], 0.1), ("original", [100, 10], 0.316227766), ("match_rms_adamw", [100, 10], 0.2)],
)
def test_adjust_lr(mode, shape, expected):
    assert _adjust_lr(0.1, mode, torch.Size(shape)) == pytest.approx(expected)


def test_muon_rejects_nonmatrix_parameters():
    with pytest.raises(ValueError, match="only supports 2D"):
        Muon([torch.ones(10)])


def test_muon_step_reduces_quadratic_loss():
    parameter = torch.nn.Parameter(torch.eye(4))
    optimizer = Muon([parameter], lr=0.1, weight_decay=0)
    loss = parameter.square().sum()
    loss.backward()
    optimizer.step()
    assert torch.isfinite(parameter).all()
    assert parameter.square().sum() < loss.detach()


@pytest.mark.parametrize("name,use_muon", [("head", True), ("head_nomuon", False)])
def test_composite_routes_parameters_and_exposes_child_groups(name, use_muon):
    from mini_trainer.training.muon import MuonAuxAdamW

    matrix = torch.nn.Parameter(torch.eye(4))
    bias = torch.nn.Parameter(torch.ones(4))
    optimizer = MuonAuxAdamW([{"name": name, "params": [matrix, bias]}], lr=0.01)
    expected = {"muon": [matrix], "adamw": [bias]} if use_muon else {"adamw": [matrix, bias]}
    assert list(optimizer.optimizers) == list(expected)
    for child_name, parameters in expected.items():
        child = getattr(optimizer, child_name)
        assert [id(p) for group in child.param_groups for p in group["params"]] == [id(p) for p in parameters]
        assert all(any(group is exposed for exposed in optimizer.param_groups) for group in child.param_groups)
    # Scheduler changes through the composite must reach every child optimizer.
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    (matrix.sum() + bias.sum()).backward()
    optimizer.step()
    scheduler.step()
    assert optimizer._step_count == 1
    assert all(getattr(optimizer, child).param_groups[0]["lr"] == 0.005 for child in expected)
