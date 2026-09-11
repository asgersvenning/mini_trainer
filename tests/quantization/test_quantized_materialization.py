import copy
import json

import pytest
import torch

from mini_trainer.export import main as export_checkpoint
from mini_trainer.hierarchical.model import HierarchicalClassifier
from mini_trainer.modeling import Classifier, classification_module
from mini_trainer.modeling.quantized_training import materialize_quantized_training_state, prepare_quantized_training
from tests.integration.test_integration_train import TinyMockModel

pytest.importorskip("torchao")
pytest.importorskip("triton")


@pytest.mark.parametrize("normalized", [False, True])
def test_materialization_preserves_represented_weights_and_source(normalized):
    torch.manual_seed(9)
    model = torch.nn.Linear(8, 3)
    if normalized:
        torch.nn.utils.parametrizations.weight_norm(model, dim=0)
    prepare_quantized_training(model)
    weight = model.parametrizations.weight.original1 if normalized else model.weight
    with torch.no_grad():
        weight.scale[0].neg_()
        weight.scale[1].zero_()
    state = model.state_dict()
    codes, scales = weight.int_data.clone(), weight.scale.clone()
    source_recipe = copy.deepcopy(state["_quantized_training"])
    values, report = materialize_quantized_training_state(state)
    restored = torch.nn.Linear(8, 3)
    if normalized:
        torch.nn.utils.parametrizations.weight_norm(restored, dim=0)
        magnitude = state["parametrizations.weight.original0"].flatten()
        expected = codes.float() * (scales.sign() * magnitude / codes.float().norm(dim=1)).view(-1, 1)
    else:
        expected = codes.float() * scales.view(-1, 1)
    restored.load_state_dict(values, strict=True)
    torch.testing.assert_close(restored.weight, expected, rtol=1e-6, atol=1e-7)
    assert torch.isfinite(restored(torch.randn(2, 8))).all()
    torch.testing.assert_close(weight.int_data, codes, rtol=0, atol=0)
    torch.testing.assert_close(weight.scale, scales, rtol=0, atol=0)
    assert state["_quantized_training"] == source_recipe
    assert not report["dynamic_activation_quantization_preserved"]
    assert not report["training_resume_supported"]
    assert not any(getattr(v, "_is_quantized_training", False) for v in values.values())
    values["bias"].zero_()
    assert not torch.equal(values["bias"], state["bias"])
    report["source_recipes"]["_quantized_training"]["quantized_modules"].append("changed")
    assert state["_quantized_training"] == source_recipe


@pytest.mark.parametrize("failure", ["missing_recipe", "wrong_module", "orphan_weight", "bad_scale", "zero_direction"])
def test_invalid_native_state_is_rejected(failure):
    model = torch.nn.utils.parametrizations.weight_norm(torch.nn.Linear(8, 3), dim=0)
    prepare_quantized_training(model)
    state = model.state_dict()
    weight = state["parametrizations.weight.original1"]
    if failure == "missing_recipe":
        state.pop("_quantized_training")
    elif failure == "wrong_module":
        state["_quantized_training"]["quantized_modules"] = ["unknown"]
    elif failure == "orphan_weight":
        state["extra.weight"] = weight
    elif failure == "bad_scale":
        weight.scale[0] = float("nan")
    else:
        weight.int_data[0].zero_()
    with pytest.raises(ValueError):
        materialize_quantized_training_state(state)


def _shared_magnitude_model():
    model = torch.nn.ModuleDict({name: torch.nn.utils.parametrizations.weight_norm(torch.nn.Linear(8, 3), dim=0) for name in ("a", "b")})
    model["b"].parametrizations.weight.original0 = model["a"].parametrizations.weight.original0
    return model


def test_shared_magnitude_with_opposite_scale_signs_preserves_both_weights():
    model = _shared_magnitude_model()
    prepare_quantized_training(model)
    model["b"].parametrizations.weight.original1.scale.neg_()
    expected = {name: model[name].weight.dequantize().detach().clone() for name in ("a", "b")}
    state, _ = materialize_quantized_training_state(model.state_dict())
    restored = _shared_magnitude_model()
    restored.load_state_dict(state, strict=True)
    for name in expected:
        torch.testing.assert_close(restored[name].weight, expected[name], rtol=1e-6, atol=1e-7)


def test_incompatible_shared_magnitude_zero_scale_fails_explicitly():
    model = _shared_magnitude_model()
    prepare_quantized_training(model)
    model["b"].parametrizations.weight.original1.scale[0].zero_()
    with pytest.raises(ValueError, match="tied parameter"):
        materialize_quantized_training_state(model.state_dict())


def test_weight_shared_between_normalized_and_ordinary_roles_fails_explicitly():
    model = torch.nn.ModuleDict(
        {"a": torch.nn.Linear(8, 3), "b": torch.nn.utils.parametrizations.weight_norm(torch.nn.Linear(8, 3), dim=0)}
    )
    model["a"].weight = model["b"].parametrizations.weight.original1
    prepare_quantized_training(model)
    with pytest.raises(ValueError, match="tied parameter"):
        materialize_quantized_training_state(model.state_dict())


@pytest.mark.parametrize("head", [Classifier, HierarchicalClassifier])
def test_masked_normalized_checkpoint_materializes_and_exports_on_cpu(tmp_path, head):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxscript")
    pytest.importorskip("onnxruntime")
    kwargs = {"sparse_masks": [torch.tensor([0, 0, 1])]} if head is HierarchicalClassifier else {}
    model, _ = head.build(model_type=TinyMockModel(), num_classes=3, hidden=True, normalized=True, **kwargs)
    classification_module(model).set_active_features([0, 2])
    prepare_quantized_training(model)
    path = tmp_path / "native.pt"
    torch.save(model.state_dict(), path)
    before = path.read_bytes()
    output = export_checkpoint(str(path), str(tmp_path / "export"), input_shape=[3, 5, 5], materialize_int8_training=True)
    assert path.read_bytes() == before
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["quantized_training_forward"] is False
    conversion = manifest["source"]["quantized_training_materialization"]
    assert len(conversion["converted_weights"]) == 2
    assert not conversion["dynamic_activation_quantization_preserved"]
    assert manifest["outputs"][0]["example_shape"][-1] == 2
    assert len(manifest["outputs"]) == (2 if head is HierarchicalClassifier else 1)
    assert manifest["verification"]["reference_device"] == "cpu"
    with pytest.raises(ValueError, match="reference_device='cuda'"):
        export_checkpoint(str(path), str(tmp_path / "native-export"), input_shape=[3, 5, 5])
