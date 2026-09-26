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
