import copy
import hashlib
import importlib.util
import json

import numpy as np
import pytest
import torch
from torchvision.models.vision_transformer import VisionTransformer

from mini_trainer.export import main as export_checkpoint
from mini_trainer.hierarchical.model import (
    AutoregressiveClassifier,
    AutoregressiveClassifierV2,
    ConditionalClassifier,
    HierarchicalClassifier,
    IndependentClassifier,
)
from mini_trainer.modeling import Classifier
from mini_trainer.modeling.onnx import export_onnx
from tests.integration.test_integration_train import TinyMockModel
from tests.training.test_checkpoint_contract import assert_state_equal

pytestmark = pytest.mark.skipif(
    any(importlib.util.find_spec(name) is None for name in ("onnx", "onnxscript", "onnxruntime")),
    reason="Install mini_trainer[export] to run ONNX integration tests",
)


def build_classifier(cls=Classifier, **kwargs):
    hierarchical = issubclass(cls, HierarchicalClassifier)
    classes = {"second": 1, "first": 0, "third": 2}
    if hierarchical:
        kwargs.update(sparse_masks=[torch.tensor([0, 0, 1])], cls2idx={"0": classes, "1": {"parent_a": 0, "parent_b": 1}})
    else:
        kwargs["cls2idx"] = classes
    if issubclass(cls, (AutoregressiveClassifier, AutoregressiveClassifierV2)):
        kwargs["decoder_kwargs"] = {"num_layers": 1, "nhead": 1, "dropout": 0.0}
    return cls.build(model_type=TinyMockModel(), num_classes=3, hidden=False, **kwargs)


@pytest.mark.parametrize("normalized,masked", [(False, False), (True, False), (True, True)])
def test_flat_export_preserves_state_and_predictions(tmp_path, normalized, masked):
    import onnxruntime as ort

    torch.manual_seed(42)
    model, _ = build_classifier(normalized=normalized, prior=[0.1, -0.2, 0.3])
    if masked:
        model.fc.set_active_features([0, 2])
    model.train()
    model.features.eval()
    modes = [module.training for module in model.modules()]
    state = copy.deepcopy(model.state_dict())
    caches = {name: value.clone() for name, value in model.named_buffers()}
    rng = torch.get_rng_state().clone()
    example = torch.linspace(-1, 1, 2 * 3 * 5 * 5).reshape(2, 3, 5, 5)
    output = export_onnx(model, example, tmp_path / "bundle")
    assert_state_equal(model.state_dict(), state)
    assert [module.training for module in model.modules()] == modes
    for name, value in model.named_buffers():
        torch.testing.assert_close(value, caches[name], rtol=0, atol=0)
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["input"]["shape"] == ["batch", 3, 5, 5]
    assert manifest["artifacts"]["model.onnx"] == hashlib.sha256((output / "model.onnx").read_bytes()).hexdigest()
    assert {case["batch_size"] for case in manifest["verification"]["cases"]} >= {1, 2, 4}
    class_mapping = manifest["classifiers"][0]["metadata"]["cls2idx"]
    assert class_mapping == ({"first": 0, "third": 1} if masked else {"first": 0, "second": 1, "third": 2})
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(output / "model.onnx"), sess_options=options, providers=["CPUExecutionProvider"])
    model.eval()
    inputs = torch.randn(3, 3, 5, 5)
    with torch.inference_mode():
        expected = model(inputs).numpy()
    np.testing.assert_allclose(session.run(None, {"images": inputs.numpy()})[0], expected, rtol=1e-4, atol=1e-5)
    with pytest.raises(FileExistsError):
        export_onnx(model, example, output)


@pytest.mark.parametrize(
    "head", [HierarchicalClassifier, ConditionalClassifier, IndependentClassifier, AutoregressiveClassifier, AutoregressiveClassifierV2]
)
def test_all_hierarchical_head_families(tmp_path, head):
    model, _ = build_classifier(head)
    model.fc.set_active_features([0, 2])
    path = export_onnx(model, torch.randn(2, 3, 5, 5), tmp_path / "bundle")
    manifest = json.loads((path / "manifest.json").read_text())
    assert len(manifest["outputs"]) == 2
    assert manifest["output_structure"] == {"list": [{"tensor": "output_0"}, {"tensor": "output_1"}]}
    assert manifest["classifiers"][0]["metadata"]["cls2idx"]["0"] == {"first": 0, "third": 1}


@pytest.mark.parametrize("backbone", ["resnet18", "efficientnet_b0", "vit"])
def test_backbone_families(tmp_path, backbone):
    if backbone == "vit":
        backbone = VisionTransformer(image_size=32, patch_size=8, num_layers=1, num_heads=2, hidden_dim=16, mlp_dim=32)
    model, _ = Classifier.build(
        model_type=backbone,
        num_classes=3,
        cls2idx={"a": 0, "b": 1, "c": 2},
        resize_size=32,
        hidden=8,
        normalized=True,
        model_args={"pretrained": False},
    )
    export_onnx(model, torch.randn(2, 3, 32, 32), tmp_path / "bundle")


class StructuredOutputs(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 3)

    def forward(self, inputs):
        logits = self.linear(inputs)
        return {"logits": logits, "extra": (logits.softmax(-1), None)}


def test_custom_outputs_wrappers_and_static_batch(tmp_path):
    model = torch.nn.DataParallel(StructuredOutputs()).double()
    output = export_onnx(model, torch.randn(1, 4), tmp_path / "bundle", dynamic_batch=False)
    assert next(model.parameters()).dtype == torch.float64
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["input"]["shape"] == [1, 4]
    assert manifest["output_structure"] == {"dict": {"logits": {"tensor": "output_0"}, "extra": {"tuple": [{"tensor": "output_1"}, None]}}}


@pytest.mark.parametrize(
    "head",
    [
        Classifier,
        HierarchicalClassifier,
        ConditionalClassifier,
        IndependentClassifier,
        AutoregressiveClassifier,
        AutoregressiveClassifierV2,
    ],
)
def test_checkpoint_export_without_downloads(tmp_path, monkeypatch, head):
    def forbidden(*args, **kwargs):
        pytest.fail("Export must not download pretrained weights")

    monkeypatch.setattr(torch.hub, "download_url_to_file", forbidden)
    model, _ = build_classifier(head)
    weights = tmp_path / "weights.pt"
    torch.save(model.state_dict(), weights)
    output = export_checkpoint(str(weights), str(tmp_path / "bundle"), input_shape=[3, 5, 5])
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["source"]["checkpoint_sha256"] == hashlib.sha256(weights.read_bytes()).hexdigest()


def test_failed_export_does_not_publish_bundle(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("Simulated exporter failure")

    model, _ = build_classifier()
    monkeypatch.setattr(torch.onnx, "export", fail)
    with pytest.raises(RuntimeError, match="Simulated exporter failure"):
        export_onnx(model, torch.randn(2, 3, 5, 5), tmp_path / "bundle")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("backend", ["timm", "transformers", "open_clip"])
def test_optional_backend_wrappers(tmp_path, backend):
    dependency = pytest.importorskip(backend)
    if backend == "timm":
        backbone = dependency.create_model("resnet18", pretrained=False)
    elif backend == "transformers":
        from mini_trainer.modeling.architectures.transformers import TransformersBackboneWrapper

        config = dependency.ViTConfig(
            image_size=32, patch_size=8, hidden_size=16, num_hidden_layers=1, num_attention_heads=2, intermediate_size=32
        )
        backbone = TransformersBackboneWrapper(dependency.ViTForImageClassification(config), "classifier")
    else:
        from mini_trainer.modeling.architectures.core import BackboneModel

        encoder = dependency.CLIP(
            embed_dim=32,
            vision_cfg={"layers": 1, "width": 32, "head_width": 16, "patch_size": 8, "image_size": 32},
            text_cfg={"context_length": 8, "vocab_size": 32, "width": 32, "heads": 2, "layers": 1},
        )
        backbone = BackboneModel(encoder, encoder_method="encode_image")
    model, _ = Classifier.build(model_type=backbone, num_classes=3, hidden=False, resize_size=32)
    export_onnx(model, torch.randn(2, 3, 32, 32), tmp_path / "bundle")


def test_parity_failure_does_not_publish_bundle(tmp_path, monkeypatch):
    import onnxruntime as ort

    real_session = ort.InferenceSession

    class IncorrectSession:
        def __init__(self, *args, **kwargs):
            self.session = real_session(*args, **kwargs)

        def run(self, *args, **kwargs):
            return [value + 1 for value in self.session.run(*args, **kwargs)]

    model, _ = build_classifier()
    monkeypatch.setattr(ort, "InferenceSession", IncorrectSession)
    with pytest.raises(AssertionError, match="ONNX parity failed"):
        export_onnx(model, torch.randn(2, 3, 5, 5), tmp_path / "bundle")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("head", [Classifier, HierarchicalClassifier])
def test_efficientnet_v2_s_symmetric_normalized_heads(tmp_path, head):
    torch.manual_seed(42)
    kwargs = {}
    if head is HierarchicalClassifier:
        kwargs["sparse_masks"] = [torch.arange(25) % 15]
    model, _ = head.build(
        model_type="efficientnet_v2_s",
        model_args={"pretrained": False},
        num_classes=25,
        hidden=True,
        normalized=True,
        **kwargs,
    )
    example = torch.randn(2, 3, 128, 128)
    destination = export_onnx(model, example, tmp_path / "efficientnet-v2", verification_inputs=[torch.randn(3, 3, 128, 128)])
    manifest = json.loads((destination / "manifest.json").read_text())
    assert {case["batch_size"] for case in manifest["verification"]["cases"]} == {1, 2, 3, 4}
    assert len(manifest["outputs"]) == (2 if head is HierarchicalClassifier else 1)
    assert manifest["classifiers"][0]["metadata"]["in_features"] == 1280
