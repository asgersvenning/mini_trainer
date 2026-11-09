import math

import pytest
import torch
from torch.nn import functional as F

from mini_trainer.hierarchical.loss import MultiLevelWeightedCrossEntropyLoss
from mini_trainer.hierarchical.model import HierarchicalClassifier

DIM, SIZES, MASK = 32, [12, 4], torch.arange(12) % 4


def outputs():
    torch.manual_seed(0)
    head = HierarchicalClassifier(
        in_features=DIM,
        out_features=SIZES[0],
        sparse_masks=[MASK],
        hidden=True,
        normalized=True,
        trainable_bias=True,
        skip_spherical_init=True,
    ).train()
    head.linear.bias.data.normal_(0, 0.5)
    species = torch.randint(0, SIZES[0], (16,))
    return head, head(torch.randn(16, DIM)), torch.stack([species, MASK[species]], 1)


def criterion(margin=0.0):
    return MultiLevelWeightedCrossEntropyLoss(
        num_classes=SIZES, device=torch.device("cpu"), dtype=torch.float32, species_margin=margin, embedding_dim=DIM if margin else None
    )


def test_margin_changes_the_species_loss_only_and_leaves_the_marginals_alone():
    head, preds, target = outputs()
    plain, margined = criterion()(preds, target), criterion(0.3)(preds, target)
    assert margined[1].item() == plain[1].item()
    assert margined[0].item() > plain[0].item()
    # The genus gradient, which is all the marginals contribute, is identical with or without the margin.
    assert margined[1].grad_fn is not None
    genus_with_margin = torch.autograd.grad(margined[1], head.linear.parametrizations.weight.original1, retain_graph=True)[0]
    genus_plain = torch.autograd.grad(plain[1], head.linear.parametrizations.weight.original1, retain_graph=True)[0]
    assert torch.equal(genus_with_margin, genus_plain)


def test_margin_is_the_arcface_rotation_of_the_true_class():
    _, preds, target = outputs()
    logits, bias = preds[0].detach(), 0.0
    var = 1 / math.sqrt(DIM - 2)
    # cos(theta + m) through lepinet's formula, then back to z-scores
    cos = torch.sin(logits * var).clamp(-1 + 1e-7, 1 - 1e-7)
    sin = torch.sqrt((1 - cos * cos).clamp_min(1e-12))
    phi = cos * math.cos(0.3) - sin * math.sin(0.3)
    one_hot = F.one_hot(target[:, 0], SIZES[0]).float()
    rotated = (torch.acos(-(one_hot * phi + (1 - one_hot) * cos)) - torch.pi / 2) / var + bias
    shifted = logits - one_hot * 0.3 * math.sqrt(DIM - 2)
    assert torch.allclose(rotated, shifted, atol=1e-2)


def test_margin_is_training_only():
    _, preds, target = outputs()
    with torch.no_grad():
        assert criterion(0.3)(preds, target)[0].item() == criterion()(preds, target)[0].item()


def test_margin_requires_the_embedding_size():
    with pytest.raises(ValueError, match="embedding_dim"):
        MultiLevelWeightedCrossEntropyLoss(num_classes=SIZES, device=torch.device("cpu"), dtype=torch.float32, species_margin=0.3)
