import importlib.util
import os
import sys
from types import ModuleType

import pytest
import torch

from mini_trainer.modeling import BackboneInfo, get_model, list_supported_backbones
from mini_trainer.modeling.architectures import load
from mini_trainer.modeling.classifier import Classifier

slow = pytest.mark.skipif(
    os.environ.get("RUN_SLOW_TESTS") != "1",
    reason="Slow architecture tests skipped by default. Set RUN_SLOW_TESTS=1 to run.",
)


@pytest.mark.parametrize("available", [False, True])
def test_list_supported_backbones(available, monkeypatch):
    import torchvision

    monkeypatch.setattr(torchvision.models, "list_models", lambda: ["shared", "tv-only"])
    monkeypatch.setattr(load, "get_bioclip_models", lambda: ["local", "blocked"])
    monkeypatch.setattr(load, "load_blacklist", lambda: {"timm": ["shared"], "bioclip": ["blocked"]})
    names = (
        "open_clip",
        "timm",
        "transformers",
        "transformers.models",
        "transformers.models.auto",
        "transformers.models.auto.configuration_auto",
        "transformers.models.auto.modeling_auto",
    )
    modules = {name: ModuleType(name) for name in names}
    modules["timm"].list_models = lambda: ["shared", "timm-only"]
    supported, unsupported = object(), object()
    modules[names[-2]].CONFIG_MAPPING = {"vit": supported, "not-an-image-model": unsupported, "swin": supported}
    modules[names[-1]].MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING = {supported: object()}
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module if available else None)

    rows = list_supported_backbones()
    assert all(isinstance(row, BackboneInfo) for row in rows)
    assert rows[:4] == [
        ("shared", "torchvision", True, False),
        ("tv-only", "torchvision", True, False),
        ("bioclip:local", "bioclip", available, False),
        ("bioclip:blocked", "bioclip", available, True),
    ]
    if available:
        assert rows[4:] == [
            ("timm:shared", "timm", True, True),
            ("timm:timm-only", "timm", True, False),
            ("hf-hub:microsoft/swin-tiny-patch4-window7-224", "transformers", True, False),
            ("hf-hub:google/vit-base-patch16-224", "transformers", True, False),
        ]
    else:
        # Missing optional libraries still advertise examples, without promising availability.
        assert {row.backend for row in rows[4:]} == {"timm", "transformers"}
        for row in rows[4:]:
            assert not row.availability and not row.blacklisted
            assert row.model.startswith("timm:" if row.backend == "timm" else "hf-hub:")


@slow
def test_torchvision_model():
    model, classifier_name, preprocess_fn, embed_dim, _ = get_model("resnet18")
    assert classifier_name == "fc"

    # Check that we can build a Classifier with it
    classifier_model, transform = Classifier.build(model_type="resnet18", num_classes=10, device="cpu")

    dummy_input = torch.randn(2, 3, 224, 224)
    preprocessed = torch.stack([transform(img) for img in dummy_input])
    outputs = classifier_model(preprocessed)

    assert outputs.shape == (2, 10)
    assert isinstance(outputs, torch.Tensor)
    assert isinstance(embed_dim, int)


has_timm = importlib.util.find_spec("timm") is not None
has_transformers = importlib.util.find_spec("transformers") is not None


@slow
@pytest.mark.skipif(not has_timm, reason="timm package not installed")
def test_timm_model():
    # Explicit prefix
    model, classifier_name, preprocess_fn, embed_dim, _ = get_model("timm:resnet18", model_args={"pretrained": False})
    # Timm resnet18 has classifier named 'fc'
    assert classifier_name == "fc"

    # Explicit prefix
    model_auto, classifier_name_auto, _, _, _ = get_model("timm:vit_tiny_patch16_224", model_args={"pretrained": False})
    assert classifier_name_auto == "head"

    classifier_model, transform = Classifier.build(
        model_type="timm:resnet18", num_classes=5, device="cpu", model_args={"pretrained": False}
    )

    dummy_input = torch.randn(2, 3, 224, 224)
    preprocessed = torch.stack([transform(img) for img in dummy_input])
    outputs = classifier_model(preprocessed)

    assert outputs.shape == (2, 5)
    assert isinstance(outputs, torch.Tensor)
    assert isinstance(embed_dim, int)


@slow
@pytest.mark.skipif(not has_transformers, reason="transformers package not installed")
def test_transformers_model():
    # Load vit model offline to avoid hitting the internet
    model_type = "google/vit-base-patch16-224"
    model, classifier_name, preprocess_fn, embed_dim, _ = get_model(
        f"transformers:{model_type}", model_args={"pretrained": False, "local_files_only": True}
    )
    assert classifier_name == "classifier"

    classifier_model, transform = Classifier.build(
        model_type=f"transformers:{model_type}",
        num_classes=7,
        device="cpu",
        model_args={"pretrained": False, "local_files_only": True},
    )

    dummy_input = torch.randn(2, 3, 224, 224)
    preprocessed = torch.stack([transform(img) for img in dummy_input])
    outputs = classifier_model(preprocessed)

    assert outputs.shape == (2, 7)
    assert isinstance(outputs, torch.Tensor)
    assert isinstance(embed_dim, int)
