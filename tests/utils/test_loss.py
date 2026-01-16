import torch

from mini_trainer.utils.loss import (
    EvenCrossEntropyLoss,
    class_weight_distribution_regularization,
    coherence_hinge_regularization,
    kl_distill_ema,
    weight_kl_gaussian,
)


def test_EvenCrossEntropyLoss():
    loss_fn = EvenCrossEntropyLoss()
    input = torch.randn(4, 10)
    target = torch.randint(0, 10, (4,))
    loss = loss_fn(input, target)
    # loss is (1,) because of broadcasting with max_CE
    assert loss.numel() == 1
    loss = loss.sum() # make scalar
    assert loss > 0
    
    # Check scaling
    # Standard CE
    ce = torch.nn.CrossEntropyLoss()(input, target)
    max_ce = torch.tensor(10.0).log() # roughly log(num_classes) if we assume max entropy?
    # Actually code uses log(num_classes)
    # max_CE = input.new_full((1, ), input.size(1), requires_grad=False).log()
    expected = ce / torch.tensor(input.size(1)).float().log()
    assert torch.allclose(loss, expected)


def test_kl_distill_ema():
    logits = torch.randn(4, 10)
    ema_logits = torch.randn(4, 10)
    loss = kl_distill_ema(logits, ema_logits, T=1.0)
    assert loss >= 0
    

def test_regularizations():
    W = torch.randn(10, 32)
    
    # class_weight_distribution_regularization
    reg1 = class_weight_distribution_regularization(W, sparse=False)
    assert reg1.numel() == 1
    
    # weight_kl_gaussian
    reg2 = weight_kl_gaussian(W, sparse=False)
    assert reg2.numel() == 1
    
    # coherence_hinge_regularization
    reg3 = coherence_hinge_regularization(W, sparse=False)
    assert reg3.numel() == 1
    
    # Test sparse
    reg4 = class_weight_distribution_regularization(W, sparse=True)
    assert reg4.numel() == 1
    
    reg5 = weight_kl_gaussian(W, sparse=True)
    assert reg5.numel() == 1
    
    reg6 = coherence_hinge_regularization(W, sparse=True)
    assert reg6.numel() == 1
