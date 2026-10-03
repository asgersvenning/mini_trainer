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


def test_release_checkpoint_identity_preserves_preprocessing(checkpoint, tmp_path, monkeypatch):
    descriptor = deepcopy(deploy._release())
    descriptor["state_sha256"] = deploy._checkpoint_digest(checkpoint)
    descriptor["preprocessing"].update(square_size=8, resize_size=10, crop_size=8)
    monkeypatch.setattr(deploy, "_release", lambda: descriptor)
    monkeypatch.setattr(deploy, "ensure_weights", lambda *args: (deepcopy(checkpoint), "fixture"))
    path = tmp_path / "renamed.pt"
    torch.save(checkpoint, path)
    predictors = [
        deploy.Predictor(device="cpu", model="full"),
        deploy.Predictor(device="cpu", model=path),
        deploy.Predictor(device="cpu", weights=checkpoint),
    ]
    image = torch.arange(192, dtype=torch.uint8).reshape(3, 8, 8)
    for p in predictors[1:]:
        assert p.preprocessing == predictors[0].preprocessing
        assert p.metadata["name"] == "Nemo"
        torch.testing.assert_close(p.preproc(image), predictors[0].preproc(image), rtol=0, atol=0)
        torch.testing.assert_close(p(image).confidence, predictors[0](image).confidence)
    changed = deepcopy(checkpoint)
    tensor = next(value for value in changed.values() if isinstance(value, torch.Tensor) and value.is_floating_point())
    tensor.add_(1)
    assert deploy._checkpoint_digest(changed) != descriptor["state_sha256"]
    resized = deepcopy(checkpoint)
    resized["fc._extra_state"]["resize_size"] = 16
    assert deploy._checkpoint_digest(resized) != descriptor["state_sha256"]
    masked = deepcopy(checkpoint)
    masked["fc.active_indices"] = torch.tensor([2])
    selected = deploy.Predictor(device="cpu", weights=masked)
    assert selected.preprocessing == predictors[0].preprocessing
    assert selected.class_list == ["c"]


@pytest.mark.parametrize("source", ["weights", "model"])
def test_cli_explicit_checkpoint_keeps_its_vocabulary(source, monkeypatch):
    captured = []
    monkeypatch.setattr(
        "mini_trainer.hierarchical.predict.cli",
        lambda **kwargs: {
            "model": "legacy.pt" if source == "model" else None,
            "weights": "legacy.pt" if source == "weights" else None,
            "class_list": None,
        },
    )
    monkeypatch.setattr(deploy, "ensure_weights", lambda *args: ("legacy.pt", "legacy.pt"))
    monkeypatch.setattr("mini_trainer.predict.main", lambda **kwargs: captured.append(kwargs))
    deploy.run()
    assert captured[0]["weights"] == "legacy.pt"
    assert captured[0]["class_list"] is None


def test_native_accepts_legacy_bfloat16_preprocessing(checkpoint):
    p = deploy.Predictor(device="cpu", weights=checkpoint)
    p.preproc = lambda x: x.to(torch.bfloat16)
    result = p(torch.zeros(3, 8, 8))
    assert torch.isfinite(result.confidence).all()


def test_cli_and_native_release_preprocessing_match(checkpoint, tmp_path, monkeypatch):
    descriptor = deepcopy(deploy._release())
    descriptor["state_sha256"] = deploy._checkpoint_digest(checkpoint)
    descriptor["preprocessing"].update(square_size=8, resize_size=10, crop_size=8)
    monkeypatch.setattr(deploy, "_release", lambda: descriptor)
    path = tmp_path / "weights.pt"
    torch.save(checkpoint, path)
    native = deploy.Predictor(device="cpu", weights=path)
    captured = []
    monkeypatch.setattr("mini_trainer.hierarchical.predict.cli", lambda **kwargs: {"model": None, "weights": str(path), "class_list": None})

    def main(**kwargs):
        captured.append(kwargs["builder"].build_model(weights=kwargs["weights"], device="cpu", dtype=torch.float32))

    monkeypatch.setattr("mini_trainer.predict.main", main)
    deploy.run()
    model, preproc = captured[0]
    model.eval()
    image = torch.arange(192, dtype=torch.uint8).reshape(1, 3, 8, 8)
    torch.testing.assert_close(preproc(image), native.preproc(image), rtol=0, atol=0)
    with torch.inference_mode():
        torch.testing.assert_close(model(preproc(image)), native.forward(native.preproc(image))[0])


def test_cli_writes_predictions_for_explicit_checkpoint(checkpoint, tmp_path, monkeypatch):
    from PIL import Image

    from mini_trainer.logging import RawResultCollector

    checkpoint["fc._extra_state"]["preprocess_dtype"] = "bfloat16"
    path = tmp_path / "checkpoint.pt"
    torch.save(checkpoint, path)
    image = tmp_path / "image.png"
    Image.new("RGB", (8, 8), (10, 20, 30)).save(image)
    monkeypatch.setattr(
        "mini_trainer.hierarchical.predict.cli",
        lambda **kwargs: {
            "model": None,
            "weights": str(path),
            "input": str(image),
            "output": str(tmp_path),
            "name": "cli",
            "class_list": None,
            "device": "cpu",
            "dtype": "float32",
            "collector_cls": RawResultCollector,
            "dataloader_builder_kwargs": {"num_workers": 0, "batch_size": 1},
        },
    )
    deploy.run()
    native = deploy.Predictor(device="cpu", weights=path)
    expected = native.forward(native.preproc(native.reader(str(image)).unsqueeze(0)))[0]
    saved = torch.load(tmp_path / "cli/predictions.pt", weights_only=False)
    torch.testing.assert_close(saved["predictions"], expected)


@pytest.mark.parametrize("complete", [True, False])
def test_cli_downloads_backbone_only_for_head_only_checkpoint(checkpoint, tmp_path, monkeypatch, complete):
    import torchvision.models

    # Exercise torchvision's actual getter; intercept its network/initialization boundary.
    checkpoint["fc._extra_state"]["backbone_class"] = "resnet18"
    if complete:
        checkpoint["backbone.weight"] = torch.ones(1)
    path = tmp_path / "weights.pt"
    torch.save(checkpoint, path)
    monkeypatch.setattr("mini_trainer.hierarchical.predict.cli", lambda **kw: {"model": None, "weights": str(path), "class_list": None})

    class ReachedBackbone(Exception):
        pass

    def build_backbone(*args, **kwargs):
        assert (kwargs["weights"] is None) == complete
        raise ReachedBackbone

    monkeypatch.setattr(torchvision.models, "get_model", build_backbone)
    monkeypatch.setattr("mini_trainer.predict.main", lambda **kw: kw["builder"].build_model(weights=kw["weights"], device="cpu"))
    with pytest.raises(ReachedBackbone):
        deploy.run()
