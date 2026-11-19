"""mt_release on a tiny timm ConvNeXt: export, real-image verification, head scoring, subsets, packaging."""

import ast
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from mini_trainer.hierarchical.model import HierarchicalClassifier
from mini_trainer.release import add_head, export_release, package_hf, standalone, verify_release
from mini_trainer.release.build import load_checkpoint, split, write_head

pytestmark = pytest.mark.skipif(
    any(importlib.util.find_spec(name) is None for name in ("onnx", "onnxscript", "onnxruntime", "safetensors", "timm")),
    reason="Install mt-trainer[export,timm] and safetensors to run release tests",
)

SPECIES = {f"s{i}": i for i in range(6)}
TREE = [torch.tensor([0, 0, 1, 1, 2, 2]), torch.tensor([0, 0, 1])]
CLS2IDX = {"0": SPECIES, "1": {"g0": 0, "g1": 1, "g2": 2}, "2": {"f0": 0, "f1": 1}}


def build(magnitude_after_angle=False, seed=0):
    torch.manual_seed(seed)
    model, _ = HierarchicalClassifier.build(
        model_type="timm:convnext_atto.d2_in1k",  # tagged, like ToL's convnext_large.dinov3_lvd1689m
        model_args={"pretrained": False},
        num_classes=6,
        sparse_masks=TREE,
        cls2idx=CLS2IDX,
        hidden=True,
        normalized=True,
        trainable_scale=True,
        trainable_bias=True,
        magnitude_after_angle=magnitude_after_angle,
        resize_method="crop",
        read_size=40,
        resize_size=32,
    )
    head = model.head.fc
    with torch.no_grad():  # learned-looking magnitudes and biases, so the formula is exercised
        head.linear.parametrizations.weight.original0.uniform_(0.5, 2.0)
        head.linear.bias.normal_()
    return model.eval()


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    root = tmp_path_factory.mktemp("release")
    weights = root / "tiny.pt"
    torch.save(build().state_dict(), weights)
    rng = np.random.default_rng(0)
    images = []
    for i, (w, h) in enumerate([(64, 48), (37, 90), (120, 120), (25, 30), (80, 41), (300, 200)]):
        path = root / f"img{i}.png"
        Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8)).save(path)
        images.append(str(path))
    return weights, images


@pytest.fixture(scope="module")
def release(checkpoint, tmp_path_factory):
    weights, _ = checkpoint
    return export_release(weights, tmp_path_factory.mktemp("out") / "release", ranks=["species", "genus", "family"], head_name="tiny")


def test_release_reproduces_mini_trainer(checkpoint, release):
    weights, images = checkpoint
    report = verify_release(release, weights, images, timing=False)
    assert report["preprocessing"]["identical_images"] == 1.0
    assert report["pass"], json.dumps(report["artifacts"], indent=1)
    for name in ("trunk.onnx", "model.onnx", "trunk.safetensors"):
        art = report["artifacts"][name]
        assert all(rank["top1_agreement"] == 1.0 for rank in art["ranks"].values()), name
        assert all(rank["max_abs_logit_diff"] < 1e-3 for rank in art["ranks"].values()), name
    assert report["artifacts"]["model.safetensors"]["pass"]  # combined file: same tensors as the separate ones
    config = json.loads((release / "config.json").read_text())
    assert config["heads"]["tiny"]["sizes"] == [6, 3, 2]
    assert json.loads((release / "heads/tiny.json").read_text())["labels"][1] == ["g0", "g1", "g2"]


@pytest.mark.parametrize("magnitude_after_angle", [False, True])
def test_score_matches_the_classifier(tmp_path, magnitude_after_angle):
    model = build(magnitude_after_angle)
    trunk, classifier = split(model)
    write_head(tmp_path, "h", classifier, ["species", "genus", "family"], "test")
    head = standalone.load_head(tmp_path / "heads/h.safetensors", tmp_path / "heads/h.json")
    features = torch.randn(5, classifier.in_features)
    with torch.no_grad():
        expected = classifier(features)
    _, ranks = standalone.score(features.numpy(), head)
    for got, want in zip(ranks, expected, strict=True):
        np.testing.assert_allclose(got, want.numpy(), rtol=1e-5, atol=1e-4)


def test_species_list_is_a_subset_of_rows(tmp_path):
    model = build()
    _, classifier = split(model)
    write_head(tmp_path, "h", classifier, ["species", "genus", "family"], "test")
    head = standalone.load_head(tmp_path / "heads/h.safetensors", tmp_path / "heads/h.json")
    rows = [0, 1, 4]
    features = torch.randn(4, classifier.in_features)
    classifier.set_active_features(rows)
    with torch.no_grad():
        expected = classifier(features)
    mapping = classifier._preprocess_metadata()["cls2idx"]
    local = standalone.subset(head, rows)
    _, ranks = standalone.score(features.numpy(), local)
    for k, (got, want) in enumerate(zip(ranks, expected, strict=True)):
        order = [mapping[str(k)][label] for label in local.labels[k]]  # compare class by class, by name
        np.testing.assert_allclose(got, want.numpy()[:, order], rtol=1e-5, atol=1e-4)
    assert local.labels == [["s0", "s1", "s4"], ["g0", "g2"], ["f0", "f1"]]


def test_add_head_requires_the_same_trunk(checkpoint, release, tmp_path):
    weights, _ = checkpoint
    add_head(weights, release, "copy")
    assert "copy" in json.loads((release / "config.json").read_text())["heads"]
    other = tmp_path / "other.pt"
    torch.save(build(seed=1).state_dict(), other)
    with pytest.raises(ValueError, match="different trunk"):
        add_head(other, release, "other")


def test_package_hf_is_ready_only_with_a_licence(checkpoint, release):
    weights, images = checkpoint
    verify_release(release, weights, images, timing=False)
    card = {"title": "Tiny", "summary": "A test release."}
    assert package_hf(release, "user/tiny", card) == ["card field missing: license"]
    assert package_hf(release, "user/tiny", {**card, "license": "mit"}) == []
    assert (release / "predict.py").read_text() == Path(standalone.__file__).read_text()
    assert "## Verification" in (release / "README.md").read_text()


def test_standalone_does_not_import_mini_trainer():
    tree = ast.parse(Path(standalone.__file__).read_text())
    modules = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    modules |= {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
    assert not any(isinstance(node, ast.ImportFrom) and node.level for node in ast.walk(tree))
    assert not {m for m in modules if m.split(".")[0] == "mini_trainer"}


def test_load_checkpoint_round_trip(checkpoint):
    weights, _ = checkpoint
    model, preprocess = load_checkpoint(weights)
    assert preprocess.transform.read_size == 40 and preprocess.transform.size == 32


def test_combined_file_check_detects_one_changed_byte(release, tmp_path):
    import shutil

    from mini_trainer.release.verify import same_tensors

    folder = tmp_path / "copy"
    shutil.copytree(release, folder)
    parts = {"trunk.": "trunk.safetensors", "head.": json.loads((folder / "config.json").read_text())["heads"]["tiny"]["file"]}
    assert same_tensors(folder, "model.safetensors", parts)
    data = bytearray((folder / "model.safetensors").read_bytes())
    data[-1] ^= 1  # the last byte belongs to the last tensor's data
    (folder / "model.safetensors").write_bytes(bytes(data))
    assert not same_tensors(folder, "model.safetensors", parts)
