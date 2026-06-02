import os

import pytest
import torch

from mini_trainer.modeling import get_model, list_supported_backbones
from mini_trainer.modeling.classifier import Classifier

# Skip all tests in this file if RUN_SLOW_TESTS is not 1
pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_SLOW_TESTS") != "1",
    reason="Slow architecture tests skipped by default. Set RUN_SLOW_TESTS=1 to run.",
)


def test_list_supported_backbones():
    from mini_trainer.modeling import BackboneInfo

    backbones = list_supported_backbones()
    assert isinstance(backbones, list)
    assert len(backbones) > 0
    for b in backbones:
        assert isinstance(b, BackboneInfo)
        # Test attribute-based access
        assert hasattr(b, "model")
        assert isinstance(b.model, str)
        assert isinstance(b.backend, str)
        assert isinstance(b.availability, bool)

        # Test tuple sequence properties (ordering)
        assert len(b) == 4
        assert b[0] == b.model
        assert b[1] == b.backend
        assert b[2] == b.availability
        assert b[3] == b.blacklisted

        # Test unpacking (guaranteed ordering)
        model, backend, availability, blacklisted = b
        assert model == b.model
        assert backend == b.backend
        assert availability == b.availability
        assert blacklisted == b.blacklisted

        # Test asdict support
        d = b._asdict()
        assert d["model"] == b.model
        assert d["backend"] == b.backend
        assert d["availability"] == b.availability
        assert d["blacklisted"] == b.blacklisted

    backends = {b.backend for b in backbones}
    assert "torchvision" in backends


def test_torchvision_model():
    model, classifier_name, preprocess_fn, embed_dim = get_model("resnet18")
    assert classifier_name == "fc"

    # Check that we can build a Classifier with it
    classifier_model, transform = Classifier.build(model_type="resnet18", num_classes=10, device="cpu")

    dummy_input = torch.randn(2, 3, 224, 224)
    preprocessed = torch.stack([transform(img) for img in dummy_input])
    outputs = classifier_model(preprocessed)

    assert outputs.shape == (2, 10)
    assert isinstance(outputs, torch.Tensor)
    assert isinstance(embed_dim, int)


def test_timm_model():
    # Explicit prefix
    model, classifier_name, preprocess_fn, embed_dim = get_model("timm:resnet18", model_args={"pretrained": False})
    # Timm resnet18 has classifier named 'fc'
    assert classifier_name == "fc"

    # Explicit prefix
    model_auto, classifier_name_auto, _, _ = get_model("timm:vit_tiny_patch16_224", model_args={"pretrained": False})
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


def test_transformers_model():
    # Load vit model offline to avoid hitting the internet
    model_type = "google/vit-base-patch16-224"
    model, classifier_name, preprocess_fn, embed_dim = get_model(
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
