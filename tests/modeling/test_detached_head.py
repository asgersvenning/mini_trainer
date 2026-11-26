"""The opt-in detached learned head: a second g and b on the main head's rows, trained without moving anything else."""

import math

import pytest
import torch
import torch.nn.functional as F

from mini_trainer.builders import BaseBuilder
from mini_trainer.hierarchical.model import HierarchicalClassifier, IndependentClassifier
from mini_trainer.modeling import AuxiliaryLogitsContext, Classifier, detached_head_as_learned
from mini_trainer.trainer import head_parameter_summary
from tests.integration.test_integration_train import TinyMockModel

CLS2IDX = {"0": {"s0": 0, "s1": 1, "s2": 2, "s3": 3}, "1": {"g0": 0, "g1": 1}}
MASKS = [torch.tensor([0, 0, 1, 1])]


def build(**kwargs):
    torch.manual_seed(0)
    model, _ = HierarchicalClassifier.build(
        model_type=TinyMockModel(), num_classes=4, sparse_masks=MASKS, cls2idx=CLS2IDX, hidden=True, **kwargs
    )
    return model


def detached(model):
    head = model.fc
    with torch.no_grad():
        head.detached_linear.scale.copy_(torch.tensor([[1.4], [0.7], [2.0], [1.1]]))
        head.detached_linear.bias.copy_(torch.tensor([0.3, -1.2, 0.0, 0.8]))
    return model


def auxiliary_output(model, x):
    with AuxiliaryLogitsContext():
        main = model(x)
        (aux,) = AuxiliaryLogitsContext.get()
    return main, aux


def test_opt_in_and_refusals():
    default = Classifier(in_features=8, out_features=3)
    assert default.detached_linear is None and "detached_learned_head" not in default.metadata
    assert not any("detached" in k for k in default.state_dict())
    with pytest.raises(ValueError, match="detached_learned_head"):
        Classifier(in_features=8, out_features=3, detached_learned_head=True, global_scale=True)
    with pytest.raises(ValueError, match="detached_learned_head"):
        Classifier(in_features=8, out_features=3, detached_learned_head=True, normalized=False)
    with pytest.raises(NotImplementedError, match="detached_learned_head"):
        IndependentClassifier(in_features=8, out_features=4, sparse_masks=MASKS, detached_learned_head=True)
    model = build(detached_learned_head=True)
    head = model.fc
    assert head.detached_linear.scale.shape == (4, 1) and torch.equal(head.detached_linear.scale, torch.ones(4, 1))
    assert torch.equal(head.detached_linear.bias, torch.zeros(4)) and head.metadata["detached_learned_head"] is True
    with torch.no_grad():  # nothing is added outside the training context
        model(torch.randn(2, 3, 5, 5))
    assert AuxiliaryLogitsContext.get() == [] and not AuxiliaryLogitsContext.active()


def test_auxiliary_logits_are_the_learned_head_formula_on_shared_rows():
    model = detached(build(detached_learned_head=True)).eval()
    head = model.fc
    x = torch.randn(6, 3, 5, 5)
    main, aux = auxiliary_output(model, x)
    embeddings = head.preclassification(model.features(x))
    rows = F.normalize(head.linear.parametrizations.weight.original1, dim=-1)
    g, b = head.detached_linear.scale.squeeze(1), head.detached_linear.bias
    expected = math.sqrt(head.preclassification_size - 2) * torch.asin((embeddings @ rows.T * g).clamp(-1 + 1e-7, 1 - 1e-7)) + b
    torch.testing.assert_close(aux[0], expected, rtol=1e-5, atol=1e-5)
    for got, want in zip(aux, head.hierarchy(expected), strict=True):  # the coarser ranks aggregate as the main head's
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)
    frozen = model(x)
    for a, c in zip(main, frozen, strict=True):  # the main head is unchanged by the second one
        torch.testing.assert_close(a, c, rtol=0, atol=0)


def test_the_auxiliary_loss_reaches_only_its_own_g_and_b():
    model = detached(build(detached_learned_head=True)).train()
    x, y = torch.randn(8, 3, 5, 5), torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
    main, aux = auxiliary_output(model, x)
    names, params = zip(*[(n, p) for n, p in model.named_parameters() if p.requires_grad], strict=True)
    own = {"fc.detached_linear.scale", "fc.detached_linear.bias"}

    aux_loss = sum(F.cross_entropy(z, t) for z, t in zip(aux, (y, MASKS[0][y]), strict=True))
    aux_grads = torch.autograd.grad(aux_loss, params, allow_unused=True)
    reached = {n for n, g in zip(names, aux_grads, strict=True) if g is not None and g.abs().sum() > 0}
    assert reached == own
    main_grads = torch.autograd.grad(F.cross_entropy(main[0], y), params, allow_unused=True)
    assert all(g is None or g.abs().sum() == 0 for n, g in zip(names, main_grads, strict=True) if n in own)
    assert main_grads[names.index("features.0.weight")].abs().sum() > 0  # the trunk learns from the main head


@pytest.mark.parametrize("decay_all", [True, False])
def test_g_and_b_are_grouped_as_a_learned_heads(decay_all):
    def group_of(model, name):
        p = dict(model.named_parameters())[name]
        groups = BaseBuilder.parameter_groups(model, 1e-3, 1e-3, 0.01, 0.01, decay_all=decay_all, head_muon=True)
        (group,) = [g for g in groups if any(q is p for q in g["params"])]
        return group["name"], group["weight_decay"]

    learned = build(trainable_scale=True, trainable_bias=True)
    second = build(detached_learned_head=True)
    assert group_of(second, "fc.detached_linear.scale") == group_of(learned, "fc.linear.parametrizations.weight.original0")
    assert group_of(second, "fc.detached_linear.bias") == group_of(learned, "fc.linear.bias")


def test_export_as_an_ordinary_learned_head_scores_like_the_second_head(tmp_path):
    model = detached(build(detached_learned_head=True)).eval()
    x = torch.randn(5, 3, 5, 5)
    _, aux = auxiliary_output(model, x)
    state = detached_head_as_learned(model.state_dict(), "fc")
    assert not any("detached" in k for k in state) and state["fc._extra_state"]["trainable_scale"] is True
    assert "detached_learned_head" not in state["fc._extra_state"]
    torch.save(state, tmp_path / "learned.pt")
    learned, _ = HierarchicalClassifier.build(weights=str(tmp_path / "learned.pt"))
    learned.eval()
    assert learned.fc.detached_linear is None and learned.fc.trainable_scale and learned.fc.trainable_bias
    with torch.no_grad():
        for got, want in zip(learned(x), aux, strict=True):
            torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)
    reloaded, _ = HierarchicalClassifier.build(weights=model.state_dict())  # the trained model itself reloads whole
    assert reloaded.fc.detached_linear is not None
    torch.testing.assert_close(reloaded.fc.detached_linear.scale, model.fc.detached_linear.scale)


def test_head_summary_names_the_trainable_magnitudes_and_biases():
    text = head_parameter_summary(detached(build(detached_learned_head=True)))
    assert "fc.detached_linear.scale min 0.7000" in text and "fc.detached_linear.bias" in text
    assert "parametrizations.weight.original0" not in text  # the frozen main head is not trainable
    assert "original0" in head_parameter_summary(build(trainable_scale=True, trainable_bias=True))
