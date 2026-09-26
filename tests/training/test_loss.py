import pytest
import torch

from mini_trainer.training.loss import EvenCrossEntropyLoss, class_weight_distribution_regularization
from mini_trainer.utils import kl_distill


@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_even_cross_entropy_uniform_predictions(reduction):
    logits = torch.zeros(3, 5, dtype=torch.float64, requires_grad=True)
    loss = EvenCrossEntropyLoss(reduction=reduction)(logits, torch.tensor([0, 2, 4]))
    expected = torch.ones(3 if reduction == "none" else 1, dtype=logits.dtype)
    if reduction == "sum":
        expected *= 3
    torch.testing.assert_close(loss, expected)
    loss.sum().backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.norm() > 0


@pytest.mark.parametrize("temperature", [1.0, 3.0])
def test_distillation_matches_teacher_kl_without_teacher_gradients(temperature):
    student = torch.tensor([[0.2, -0.8], [1.1, 0.4]], requires_grad=True)
    teacher = torch.tensor([[1.4, -0.1], [-0.5, 0.8]], requires_grad=True)
    expected = (
        torch.distributions.kl_divergence(
            torch.distributions.Categorical(logits=teacher.detach() / temperature),
            torch.distributions.Categorical(logits=student.detach() / temperature),
        ).mean()
        * temperature**2
    )
    loss = kl_distill(student, teacher, T=temperature)
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert teacher.grad is None
    assert torch.isfinite(student.grad).all() and student.grad.norm() > 0


@pytest.mark.parametrize("sparse", [False, True])
def test_regularization_penalizes_clustered_directions(sparse):
    # 64 classes exceed the sampling threshold; 96 dimensions allow orthogonality.
    orthogonal = torch.eye(64, 96)
    torch.testing.assert_close(class_weight_distribution_regularization(orthogonal, sparse=sparse), torch.tensor(0.0))
    clustered = (orthogonal + 1).requires_grad_()
    penalty = class_weight_distribution_regularization(clustered, sparse=sparse)
    assert penalty.ndim == 0 and torch.isfinite(penalty) and penalty > 0
    penalty.backward()
    assert torch.isfinite(clustered.grad).all() and clustered.grad.norm() > 0
