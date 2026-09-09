import torch

from mini_trainer.training.loss import EvenCrossEntropyLoss, class_weight_distribution_regularization
from mini_trainer.utils import kl_distill


def test_EvenCrossEntropyLoss():
    loss_fn = EvenCrossEntropyLoss()
    input = torch.randn(4, 10)
    target = torch.randint(0, 10, (4,))
    loss = loss_fn(input, target)
    # loss is (1,) because of broadcasting with max_CE
    assert loss.numel() == 1
    loss = loss.sum()  # make scalar
    assert loss > 0

    # Check scaling
    # Standard CE
    ce = torch.nn.CrossEntropyLoss()(input, target)
    expected = ce / torch.tensor(input.size(1)).float().log()
    assert torch.allclose(loss, expected)


def test_kl_distill():
    logits = torch.randn(4, 10)
    ema_logits = torch.randn(4, 10)
    loss = kl_distill(logits, ema_logits, T=1.0)
    assert loss >= 0


def test_regularizations():
    W = torch.randn(10, 32)

    # class_weight_distribution_regularization
    reg1 = class_weight_distribution_regularization(W, sparse=False)
    assert reg1.numel() == 1

    # Test sparse
    reg4 = class_weight_distribution_regularization(W, sparse=True)
    assert reg4.numel() == 1
