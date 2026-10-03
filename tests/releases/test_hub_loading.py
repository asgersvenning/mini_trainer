"""Pinned Hub transport, lazy profile retrieval and validated cache links."""

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from deployment.mambo_deploy import Predictor
from deployment.mambo_deploy.hub import load_bundle
from deployment.mambo_deploy.preprocessing import RECIPE


@pytest.fixture
def hub(tmp_path, monkeypatch):
    revision = "a" * 40
    repo = tmp_path / "models--owner--nemo"
    root = repo / "snapshots" / revision
    payloads = {
        "classes.json": json.dumps({"labels": [["a"], ["g"], ["f"]], "parents": [[0], [0]]}),
        "preprocessing.json": json.dumps(RECIPE),
        "presets.json": "{}",
        "torch.pt": "torch",
        "onnx/model.onnx": "graph",
        "onnx/model.onnx.data": "external",
        "embedding/model.onnx": "embedding",
        "embedding/model.onnx.data": "vectors",
    }
    origins = {name: "https://unused.invalid/" + name for name in payloads if not name.endswith(".json")}
    manifest = {
        "schema": "mambo-release-v1",
        "model_id": "MAMBO_v3",
        "score_semantics": "hierarchical-leaf-logits-logsumexp-v1",
        "embedding": {"dimension": 1280},
        "origins": origins,
        "profiles": {
            "torch": {"model": "torch.pt", "files": ["torch.pt"]},
            "onnx": {"model": "onnx/model.onnx", "files": ["onnx/model.onnx", "onnx/model.onnx.data"]},
            "onnx-embedding": {"model": "embedding/model.onnx", "files": ["embedding/model.onnx", "embedding/model.onnx.data"]},
        },
        "files": {name: {"sha256": hashlib.sha256(data.encode()).hexdigest(), "size": len(data)} for name, data in payloads.items()},
    }
    payloads["release.json"] = json.dumps(manifest)
    calls = []

    def download(repo_id, filename, **options):
        calls.append((filename, options))
        relative = filename.removeprefix("bundle/")
        destination = root / filename
        if options.get("local_files_only") and not destination.exists():
            raise FileNotFoundError(filename)
        if not destination.exists():
            blob = repo.parent / "blobs" / hashlib.sha256(payloads[relative].encode()).hexdigest()
            blob.parent.mkdir(parents=True, exist_ok=True)
            blob.write_text(payloads[relative])
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.symlink_to(blob)
        return str(destination)

    def snapshot(repo_id, *, allow_patterns, **options):
        for filename in allow_patterns:
            download(repo_id, filename, **options)
        return str(root)

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=download, snapshot_download=snapshot))
    return root, revision, calls


def test_hub_pins_revision_and_fetches_only_requested_backend(hub):
    root, revision, calls = hub
    bundle = load_bundle("owner/nemo", backend="onnx", embeddings=False, revision="main", token="test-token", cache_dir="cache")
    assert bundle.profile("onnx").read_text() == "graph"
    assert not bundle.profile("onnx").is_symlink()
    assert not (root / "bundle/torch.pt").exists()
    assert not (root / "bundle/embedding/model.onnx").exists()
    assert all(options["revision"] == revision for _, options in calls[1:])
    assert all(options["token"] == "test-token" for _, options in calls)
    assert bundle.profile("onnx-embedding").read_text() == "embedding"
    assert (root / "bundle/embedding/model.onnx.data").exists()
    offline = load_bundle("owner/nemo", backend="onnx", embeddings=True, revision=revision, local_files_only=True)
    assert offline.profile("onnx-embedding").read_text() == "embedding"


def test_hub_offline_missing_and_corrupt_assets(hub):
    root, revision, _ = hub
    with pytest.raises(FileNotFoundError):
        load_bundle("owner/nemo", backend="torch", embeddings=False, revision=revision, local_files_only=True)
    load_bundle("owner/nemo", backend="torch", embeddings=False)
    (root / "bundle/torch.pt").write_text("bad")
    bundle = load_bundle("owner/nemo", backend="torch", embeddings=False)
    with pytest.raises(ValueError, match="mismatch"):
        bundle.profile("torch")


def test_local_snapshot_uses_same_loader_without_hub_dependency(hub, monkeypatch):
    root, _, _ = hub
    bundle = load_bundle("owner/nemo", backend="onnx", embeddings=True)
    # A normal local export contains files rather than Hub symlinks.
    import shutil

    export = root.parent.parent / "export"
    shutil.copytree(bundle.root, export)
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    loaded = []
    monkeypatch.setattr(Predictor, "load", lambda self, **kwargs: loaded.append(kwargs) or self)
    p = Predictor.from_pretrained(export, embeddings=True)
    assert p.input_size == 384 and loaded == [{"embeddings": True}]


def test_rendered_card_and_native_bootstrap_match_bundle():
    from dev.releases.mambo_v3.package_download_metadata import model_card, native_descriptor

    root = Path(__file__).parents[2]
    metadata = json.loads((root / "deployment/mambo_deploy/default_bundle.json").read_text())["metadata"]
    assert native_descriptor(metadata) == json.loads((root / "mini_trainer/nemo.json").read_text())
    manifest, classes = json.loads(metadata["release.json"]), json.loads(metadata["classes.json"])
    content = model_card(manifest, classes)
    hub = pytest.importorskip("huggingface_hub")
    card = hub.ModelCard(content)
    assert card.data.license == "cc-by-nc-sa-4.0"
    assert card.data.library_name == "mambo-deploy" and card.data.model_name == "Nemo"
    assert "{{" not in content and f"{len(classes['labels'][0]):,} species" in content


def test_hub_selection_does_not_depend_on_origins(hub):
    root, _, calls = hub
    load_bundle("owner/nemo", backend="onnx", embeddings=False)
    manifest_path = root / "bundle/release.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.pop("origins")
    manifest_path.write_text(json.dumps(manifest))
    calls.clear()
    load_bundle("owner/nemo", backend="onnx", embeddings=False)
    requested = {name for name, _ in calls}
    assert "bundle/torch.pt" not in requested
    assert "bundle/embedding/model.onnx" not in requested
