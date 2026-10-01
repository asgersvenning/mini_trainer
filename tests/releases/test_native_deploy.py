"""Native startup and the ACS public metadata contract, without portable imports."""

import sys
from copy import deepcopy

import pytest
import torch
from torchvision.transforms import v2

from mini_trainer import deploy
from mini_trainer.hierarchical.model import HierarchicalClassifier, HierarchicalPrediction


class Backbone(torch.nn.Module):
    default_transform = v2.Compose([v2.ToDtype(torch.float32, scale=True)])

    def __init__(self):
        super().__init__()
        self.pool = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten())
        self.fc = torch.nn.Linear(3, 3)

    def forward(self, x):
        return self.fc(self.pool(x))


@pytest.fixture
def checkpoint():
    model, _ = HierarchicalClassifier.build(
        model_type=Backbone(),
        num_classes=3,
        resize_size=8,
        hidden=False,
        sparse_masks=[torch.tensor([0, 0, 1]), torch.tensor([0, 0])],
        cls2idx={"0": {"a": 0, "b": 1, "c": 2}, "1": {"g0": 0, "g1": 1}, "2": {"f0": 0}},
    )
    state = model.state_dict()
    state["fc._extra_state"]["backbone_class"] = "tests.releases.test_native_deploy:Backbone"
    return state


def test_native_checkpoint_and_acs_metadata_without_portable(checkpoint, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "mambo_deploy", None)
    path = tmp_path / "weights.pt"
    torch.save(checkpoint, path)
    first = deploy.Predictor(device="cpu", weights=checkpoint)
    second = deploy.Predictor(device="cpu", model=path)
    assert first.input_size == first.resize_size == 8
    assert first.embedding_dim == 3
    assert first.model is not None and callable(first.preproc)
    assert first.classes == [["a", "b", "c"], ["g0", "g1"], ["f0"]]
    metadata = first.metadata
    metadata["cls2idx"]["0"].clear()
    assert first.cls2idx["0"] == {"a": 0, "b": 1, "c": 2}
    image = torch.arange(3 * 8 * 8, dtype=torch.uint8).reshape(3, 8, 8)
    result, vectors = first.predict_with_embeddings(image)
    assert isinstance(result, HierarchicalPrediction) and vectors.shape == (1, 3)
    torch.testing.assert_close(result.confidence, second(image).confidence)
    assert first.load() is first
    first._apply_class_mask(["c"])
    assert first.class_list == ["c"] and first(image)[0].label[0] == "c"
    assert first.classes[0] == ["a", "b", "c"]
    first._apply_class_mask(-1)
    assert first.class_list == ["a", "b", "c"]


def test_native_default_scope_and_weight_directory(checkpoint, monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(deploy, "ensure_weights", lambda model, directory: (calls.append((model, directory)) or checkpoint, "fixture"))
    descriptor = deepcopy(deploy._release())
    descriptor["presets"] = {"europe": ["a", "b"]}
    monkeypatch.setattr(deploy, "_release", lambda: descriptor)
    predictor = deploy.Predictor(device="cpu", weight_dir=tmp_path)
    assert calls == [(None, tmp_path)]
    assert predictor.preset == "europe" and predictor.class_list == ["a", "b"]
    assert predictor.metadata["model_id"] == "MAMBO_v3"


def test_saved_mask_does_not_replace_full_vocabulary(checkpoint):
    checkpoint["fc.active_indices"] = torch.tensor([2])
    p = deploy.Predictor(device="cpu", weights=checkpoint)
    assert p.classes[0] == ["a", "b", "c"] and p.class_list == ["c"]
    p._apply_class_mask(-1)
    assert p.class_list == ["a", "b", "c"]


def test_head_only_checkpoint_requests_pretrained_backbone(checkpoint, monkeypatch):
    from mini_trainer.builders import BaseBuilder

    model, preproc = BaseBuilder.build_model(weights=deepcopy(checkpoint))
    legacy = {key: value for key, value in deepcopy(checkpoint).items() if key.startswith("fc.")}
    legacy["fc._extra_state"]["backbone_class"] = "bioclip-2"
    calls = []

    def build(**kwargs):
        calls.append(kwargs)
        return model, preproc

    monkeypatch.setattr(BaseBuilder, "build_model", build)
    deploy.Predictor(device="cpu", weights=legacy)
    assert calls[0]["model_args"]["pretrained"] is True
