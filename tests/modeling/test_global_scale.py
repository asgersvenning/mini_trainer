"""The cosine head's opt-in shared learned scale: z_k = s * sqrt(D - 2) * asin(cos_k) + b_k with one scalar s."""

import importlib.util
import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from mini_trainer.builders import BaseBuilder
from mini_trainer.hierarchical.model import HierarchicalClassifier, IndependentClassifier
from mini_trainer.modeling import Classifier
from tests.integration.test_integration_train import TinyMockModel

CLS2IDX = {"0": {"s0": 0, "s1": 1, "s2": 2, "s3": 3}, "1": {"g0": 0, "g1": 1}}
MASKS = [torch.tensor([0, 0, 1, 1])]


def build(scale=1.0, **kwargs):
    torch.manual_seed(0)
    model, _ = HierarchicalClassifier.build(
        model_type=TinyMockModel(), num_classes=4, sparse_masks=MASKS, cls2idx=CLS2IDX, hidden=False, global_scale=True, **kwargs
    )
    with torch.no_grad():
        model.fc.global_scale.fill_(scale)
    return model.eval()


def test_one_shared_scale_trains_and_gives_the_tied_formula():
    model = build().train()
    head = model.fc
    assert [name for name, p in head.named_parameters() if p.requires_grad and p.numel() == 1] == ["global_scale"]
    assert head.global_scale.shape == (1,) and head.global_scale.item() == 1.0
    weight = head.linear.parametrizations.weight
    assert not weight.original0.requires_grad and not head.linear.bias.requires_grad  # g = 1 and b = 0, frozen
    assert head.metadata["global_scale"] is True

    with torch.no_grad():
        head.global_scale.fill_(2.5)
    x = torch.randn(6, head.in_features)
    logits = head(x)[0]
    cos = F.normalize(head.preclassification(x), dim=-1) @ F.normalize(weight.original1, dim=-1).T
    expected = 2.5 * math.sqrt(head.preclassification_size - 2) * torch.asin(cos.clamp(-1 + 1e-7, 1 - 1e-7))
    torch.testing.assert_close(logits, expected, rtol=1e-5, atol=1e-5)

    F.cross_entropy(logits, torch.tensor([0, 1, 2, 3, 0, 1])).backward()
    assert head.global_scale.grad is not None and head.global_scale.grad.abs().item() > 0
    assert weight.original1.grad is not None and weight.original0.grad is None


def test_the_scale_is_opt_in_and_refused_where_it_is_ambiguous():
    default = Classifier(in_features=8, out_features=3)
    assert default.global_scale is None and "global_scale" not in default.metadata
    assert not any(k.endswith("global_scale") for k in default.state_dict())
    with pytest.raises(ValueError, match="global_scale"):
        Classifier(in_features=8, out_features=3, global_scale=True, trainable_scale=True)
    with pytest.raises(ValueError, match="global_scale"):
        Classifier(in_features=8, out_features=3, global_scale=True, normalized=False)
    with pytest.raises(NotImplementedError, match="global_scale"):
        IndependentClassifier(in_features=8, out_features=4, sparse_masks=MASKS, global_scale=True)


def test_the_scale_gets_no_weight_decay_and_no_muon():
    model = build()
    groups = BaseBuilder.parameter_groups(model, head_lr=1e-3, backbone_lr=1e-3, head_weight_decay=0.1, backbone_weight_decay=0.1)
    (group,) = [g for g in groups if any(p is model.fc.global_scale for p in g["params"])]
    assert group["weight_decay"] == 0 and "nomuon" in group["name"]


def test_reload_and_class_masks_keep_the_logits(tmp_path):
    model = build(scale=1.7)
    torch.save(model.state_dict(), tmp_path / "weights.pt")
    reloaded, _ = HierarchicalClassifier.build(weights=str(tmp_path / "weights.pt"))
    reloaded.eval()
    assert reloaded.fc.global_scale.item() == pytest.approx(1.7)
    x = torch.randn(3, 3, 5, 5)
    with torch.no_grad():
        for a, b in zip(model(x), reloaded(x), strict=True):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        full = model(x)[0]
        model.fc.set_active_features([1, 3])  # a species list
        torch.testing.assert_close(model(x)[0], full[:, [1, 3]], rtol=1e-6, atol=1e-6)


def test_release_head_scores_like_the_classifier(tmp_path):
    pytest.importorskip("safetensors")
    from mini_trainer.release import standalone
    from mini_trainer.release.build import write_head

    head = build(scale=2.2).fc
    write_head(tmp_path, "h", head, ["species", "genus"], "test")
    released = standalone.load_head(tmp_path / "heads/h.safetensors", tmp_path / "heads/h.json")
    features = torch.randn(5, head.in_features)
    with torch.no_grad():
        expected = head(features)
    _, ranks = standalone.score(features.numpy(), released)
    for got, want in zip(ranks, expected, strict=True):
        np.testing.assert_allclose(got, want.numpy(), rtol=1e-5, atol=1e-4)


@pytest.mark.skipif(
    any(importlib.util.find_spec(name) is None for name in ("onnx", "onnxscript", "onnxruntime")),
    reason="Install mt-trainer[export] to run ONNX integration tests",
)
def test_onnx_export_keeps_the_logits(tmp_path):
    import onnxruntime as ort

    from mini_trainer.modeling.onnx import export_onnx

    model = build(scale=2.2)
    output = export_onnx(model, torch.linspace(-1, 1, 2 * 3 * 5 * 5).reshape(2, 3, 5, 5), tmp_path / "bundle")
    session = ort.InferenceSession(str(output / "model.onnx"), providers=["CPUExecutionProvider"])
    inputs = torch.randn(3, 3, 5, 5)
    with torch.inference_mode():
        expected = [o.numpy() for o in model(inputs)]
    for got, want in zip(session.run(None, {"images": inputs.numpy()}), expected, strict=True):
        np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-5)
