"""Native CUDA QT forward to portable integer ONNX, without float substitution."""

import json
import os

import pytest
import torch

from mini_trainer.export import main as export_checkpoint
from mini_trainer.hierarchical.model import HierarchicalClassifier
from mini_trainer.modeling import Classifier, classification_module
from mini_trainer.modeling.onnx import export_onnx
from mini_trainer.modeling.quantized_training import prepare_quantized_training
from tests.integration.test_integration_train import TinyMockModel

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
pytest.importorskip("onnxscript")
pytest.importorskip("torchao")
pytest.importorskip("triton")


def cuda():
    if os.environ.get("RUN_CUDA_TESTS") != "1":
        pytest.skip("Set RUN_CUDA_TESTS=1 for native INT8 ONNX parity")
    if not torch.cuda.is_available():
        pytest.fail("CUDA requested but unavailable")
    return torch.device("cuda:0")


def test_native_export_requires_explicit_cuda_reference(tmp_path):
    model = torch.nn.Linear(8, 3)
    prepare_quantized_training(model)
    with pytest.raises(ValueError, match="reference_device='cuda'"):
        export_onnx(model, torch.ones(2, 8), tmp_path / "bundle")
    assert not (tmp_path / "bundle").exists()


@pytest.mark.parametrize("backbone", ["tiny", "efficientnet_v2_s"])
@pytest.mark.parametrize("head", [Classifier, HierarchicalClassifier])
def test_normalized_symmetric_native_heads_export(tmp_path, monkeypatch, head, backbone):
    device = cuda()
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", True)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", True)
    torch.manual_seed(42)
    kwargs = {}
    if head is HierarchicalClassifier:
        kwargs["sparse_masks"] = [torch.tensor([0, 0, 1])]
    model, _ = head.build(
        model_type=TinyMockModel() if backbone == "tiny" else backbone,
        num_classes=3,
        hidden=True,
        normalized=True,
        model_args={"pretrained": False},
        device=device,
        **kwargs,
    )
    prepare_quantized_training(model)
    classification_module(model).set_active_features([0, 2])
    model.train()
    parameters = list(model.parameters())
    before = [(p.requires_grad, p.int_data.clone(), p.scale.clone()) for p in parameters if hasattr(p, "int_data")]
    sample = torch.randn(2, 3, 32, 32)
    path = export_onnx(model, sample, tmp_path / "bundle", reference_device=device, verification_inputs=[torch.randn(3, 3, 32, 32)])
    assert model.training
    assert torch.backends.cudnn.allow_tf32
    assert torch.backends.cuda.matmul.allow_tf32
    for p, (requires_grad, codes, scales) in zip([p for p in parameters if hasattr(p, "int_data")], before, strict=True):
        assert p.requires_grad == requires_grad
        torch.testing.assert_close(p.int_data, codes, rtol=0, atol=0)
        torch.testing.assert_close(p.scale, scales, rtol=0, atol=0)
    manifest = json.loads((path / "manifest.json").read_text())
    assert manifest["quantized_training_forward"] is True
    assert manifest["verification"]["reference_device"] == "cuda:0"
    assert {case["batch_size"] for case in manifest["verification"]["cases"]} == {1, 2, 3, 4}
    assert len(manifest["outputs"]) == (2 if head is HierarchicalClassifier else 1)
    graph = onnx.load(path / "model.onnx")
    assert sum(node.op_type == "MatMulInteger" for node in graph.graph.node) == 2


def test_native_checkpoint_cli_export(tmp_path):
    cuda()
    model, _ = Classifier.build(model_type=TinyMockModel(), num_classes=3, hidden=True, normalized=True, model_args={"pretrained": False})
    prepare_quantized_training(model)
    weights = tmp_path / "weights.pt"
    torch.save(model.state_dict(), weights)
    safe_globals = list(torch.serialization.get_safe_globals())
    path = export_checkpoint(str(weights), str(tmp_path / "bundle"), input_shape=[3, 5, 5], reference_device="cuda:0")
    assert torch.serialization.get_safe_globals() == safe_globals
    assert json.loads((path / "manifest.json").read_text())["quantized_training_forward"] is True


def test_native_linear_zero_tiny_and_negative_scale_rows(tmp_path):
    device = cuda()
    torch.manual_seed(42)
    model = torch.nn.Linear(17, 5).to(device)
    prepare_quantized_training(model)
    with torch.no_grad():
        model.weight.scale[0].neg_()
        model.weight.scale[1].zero_()
    inputs = torch.randn(2, 3, 17)
    inputs[0, 0].zero_()
    inputs[0, 1].mul_(1e-15)
    inputs[1, 0].mul_(1e5)
    with torch.autocast("cuda", dtype=torch.float16):
        path = export_onnx(model, inputs, tmp_path / "bundle", reference_device=device)
        assert torch.is_autocast_enabled("cuda")
    manifest = json.loads((path / "manifest.json").read_text())
    assert manifest["outputs"][0]["dtype"] == "float32"


def test_wide_linear_onnx_avoids_int32_saturation(tmp_path):
    device = cuda()
    model = torch.nn.Linear(150000, 2, bias=False).to(device)
    with torch.no_grad():
        model.weight.fill_(1)
    prepare_quantized_training(model)
    sample = torch.ones(2, 150000)
    path = export_onnx(model, sample, tmp_path / "bundle", reference_device=device)
    graph = onnx.load(path / "model.onnx")
    assert sum(node.op_type == "MatMulInteger" for node in graph.graph.node) > 1
    with torch.inference_mode():
        actual = model(sample.to(device))
    torch.testing.assert_close(actual, torch.full((2, 2), 150000.0, device=device), rtol=1e-6, atol=1e-3)
