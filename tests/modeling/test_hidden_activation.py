"""The opt-in hidden-layer activation: leaky ReLU by default (unchanged), or none (a linear hidden layer)."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from mini_trainer.hierarchical.model import HierarchicalClassifier
from mini_trainer.modeling import Classifier
from tests.integration.test_integration_train import TinyMockModel

CLS2IDX = {"0": {"s0": 0, "s1": 1, "s2": 2, "s3": 3}, "1": {"g0": 0, "g1": 1}}
MASKS = [torch.tensor([0, 0, 1, 1])]


def head(**kwargs):
    torch.manual_seed(0)
    return Classifier(in_features=8, out_features=5, hidden=True, **kwargs).eval()


def build(**kwargs):
    torch.manual_seed(0)
    model, _ = HierarchicalClassifier.build(
        model_type=TinyMockModel(), num_classes=4, sparse_masks=MASKS, cls2idx=CLS2IDX, hidden=True, **kwargs
    )
    return model.eval()


def test_the_default_is_bit_identical():
    default, explicit = head(), head(hidden_activation="leaky_relu")
    assert default.hidden_activation == "leaky_relu" and "hidden_activation" not in default.metadata
    assert default.metadata == explicit.metadata
    assert default.state_dict().keys() == explicit.state_dict().keys()
    x = torch.randn(7, 8)
    with torch.no_grad():
        manual = F.normalize(F.leaky_relu(default.hidden(x)), 2, -1)  # the code as it was before the option
        torch.testing.assert_close(default.preclassification(x), manual, rtol=0, atol=0)
        torch.testing.assert_close(default(x), explicit(x), rtol=0, atol=0)


def test_none_is_a_linear_hidden_layer_and_is_recorded():
    linear = head(hidden_activation="none")
    assert linear.metadata["hidden_activation"] == "none"
    x = torch.randn(7, 8)
    with torch.no_grad():
        expected = F.normalize(linear.hidden(x), 2, -1)
        torch.testing.assert_close(linear.preclassification(x), expected, rtol=0, atol=0)
        assert (linear.preclassification(x) < 0).float().mean() > 0.2  # signed codes, not the positive orthant
    with pytest.raises(ValueError, match="hidden_activation"):
        head(hidden_activation="relu")


def test_reload_rebuilds_the_activation(tmp_path):
    model = build(hidden_activation="none")
    torch.save(model.state_dict(), tmp_path / "weights.pt")
    reloaded, _ = HierarchicalClassifier.build(weights=str(tmp_path / "weights.pt"))
    reloaded.eval()
    assert reloaded.fc.hidden_activation == "none"
    x = torch.randn(3, 3, 5, 5)
    with torch.no_grad():
        for a, b in zip(model(x), reloaded(x), strict=True):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    default, _ = HierarchicalClassifier.build(weights=build().state_dict())
    assert default.fc.hidden_activation == "leaky_relu"


@pytest.mark.parametrize("activation", ["leaky_relu", "none"])
def test_release_head_scores_like_the_classifier(tmp_path, activation):
    pytest.importorskip("safetensors")
    from mini_trainer.release import standalone
    from mini_trainer.release.build import write_head

    classifier = build(hidden_activation=activation).fc
    write_head(tmp_path, "h", classifier, ["species", "genus"], "test")
    released = standalone.load_head(tmp_path / "heads/h.safetensors", tmp_path / "heads/h.json")
    assert released.hidden_activation == activation
    features = torch.randn(5, classifier.in_features)
    with torch.no_grad():
        expected = classifier(features)
    _, ranks = standalone.score(features.numpy(), released)
    for got, want in zip(ranks, expected, strict=True):
        np.testing.assert_allclose(got, want.numpy(), rtol=1e-5, atol=1e-4)
