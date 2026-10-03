"""Publication preserves reviewed payloads and pins actual Hub upload commits."""

import json
import sys
from types import SimpleNamespace

import pytest

from dev.releases.mambo_v3.prepare_candidate import HERE, digest
from dev.releases.mambo_v3.publish_assets import hub, verified_payload


@pytest.fixture
def payload(tmp_path):
    folder = tmp_path / "payload"
    folder.mkdir()
    (folder / "README.md").write_text("Public model card")
    data = {
        "source_commit": "a" * 40,
        "package_version": "0.3.0",
        "model_id": "MAMBO_v3",
        "files": {"README.md": digest(folder / "README.md")},
    }
    (folder / "publication.json").write_text(json.dumps(data))
    return folder


def test_modified_or_uninventoried_file_cannot_be_published(payload):
    verified_payload(payload)
    (payload / "README.md").write_text("Unreviewed change")
    with pytest.raises(ValueError, match="integrity"):
        verified_payload(payload)
    (payload / "private.txt").write_text("Not a release input")
    with pytest.raises(ValueError, match="file set"):
        verified_payload(payload)


@pytest.fixture
def remote(payload, monkeypatch):
    class MissingManifest(Exception):
        pass

    state = SimpleNamespace(manifest=json.loads((payload / "publication.json").read_text()), calls=[], failure=None)

    def download(*a, **kw):
        assert kw["revision"] == "b" * 40
        if state.failure:
            raise state.failure
        if state.manifest is None:
            raise MissingManifest
        path = payload.parent / "remote.json"
        path.write_text(json.dumps(state.manifest))
        return str(path)

    def upload(**kw):
        state.calls.append(kw)
        return SimpleNamespace(oid="c" * 40)

    # Deliberately no create_tag method: trusted publication does not need one.
    api = SimpleNamespace(token="unused", repo_info=lambda *a, **kw: SimpleNamespace(sha="b" * 40), upload_folder=upload)
    monkeypatch.setenv("HF_TOKEN", "unused")
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=lambda **kw: api, hf_hub_download=download))
    monkeypatch.setitem(sys.modules, "huggingface_hub.errors", SimpleNamespace(EntryNotFoundError=MissingManifest))
    return state


@pytest.mark.parametrize("kind", ["model", "space"])
def test_hub_retry_reuses_uploaded_commit_and_records_receipt(payload, remote, kind):
    receipt = hub(payload, "owner/model", kind)
    assert remote.calls == []
    assert receipt == {
        "repository": "owner/model",
        "repo_type": kind,
        "revision": "b" * 40,
        "source_commit": "a" * 40,
        "model_id": "MAMBO_v3",
        "package_version": "0.3.0",
        "publication_sha256": digest(payload / "publication.json"),
    }


@pytest.mark.parametrize("kind", ["model", "space"])
def test_hub_new_upload_uses_observed_parent_and_pins_returned_commit(payload, remote, kind):
    remote.manifest = None
    receipt = hub(payload, "owner/model", kind)
    assert receipt["revision"] == "c" * 40
    assert remote.calls[0]["parent_commit"] == "b" * 40


@pytest.mark.parametrize("kind", ["model", "space"])
def test_hub_rejects_changed_payload_at_same_release_source(payload, remote, kind):
    remote.manifest["files"] = {}
    with pytest.raises(ValueError, match="Refusing to replace"):
        hub(payload, "owner/model", kind)
    assert remote.calls == []


def test_model_cannot_be_replaced_but_reviewed_space_can_advance(payload, remote):
    remote.manifest["source_commit"] = "d" * 40
    with pytest.raises(ValueError, match="Refusing to replace"):
        hub(payload, "owner/model", "model")
    assert remote.calls == []
    assert hub(payload, "owner/model", "space")["revision"] == "c" * 40


def test_auth_failure_is_not_treated_as_missing_manifest(payload, remote):
    remote.failure = PermissionError("401 Unauthorized")
    with pytest.raises(PermissionError):
        hub(payload, "owner/model", "model")
    assert remote.calls == []


def test_maintenance_requires_new_version_and_identical_model_assets():
    from copy import deepcopy

    from dev.releases.mambo_v3.publish_assets import maintenance_update

    old = {
        "model_id": "MAMBO_v3",
        "source_commit": "a" * 40,
        "package_version": "0.3.0",
        "files": {"bundle/models/pytorch/best.pt": "weights", "bundle/classes.json": "vocabulary", "README.md": "old"},
    }
    new = deepcopy(old)
    new.update(source_commit="b" * 40, package_version="0.3.1")
    new["files"]["README.md"] = "Nemo"
    assert maintenance_update(old, new)
    for key in ("bundle/models/pytorch/best.pt", "bundle/classes.json"):
        changed = deepcopy(new)
        changed["files"][key] = "changed"
        assert not maintenance_update(old, changed)
    new["package_version"] = "0.3.0"
    assert not maintenance_update(old, new)


def test_reviewed_card_accepts_filtered_species_comparison_and_checks_figures(tmp_path):
    import shutil

    from dev.releases.mambo_v3.package_download_metadata import card_performance

    source = HERE / "card-performance"
    shutil.copytree(source, tmp_path, dirs_exist_ok=True)
    caption = card_performance(tmp_path, required=True)
    assert "941 adult-screened images" in caption
    assert "890 reporting images" in caption
    (tmp_path / "quality.png").write_bytes(b"unreviewed figure")
    with pytest.raises(ValueError, match="Changed card figure"):
        card_performance(tmp_path, required=True)


@pytest.mark.parametrize("problem", ["missing", "smoke", "filter", "split", "taxonomy", "model", "threshold"])
def test_card_rejects_incomplete_or_inconsistent_evidence(tmp_path, problem):
    from dev.releases.mambo_v3.package_download_metadata import card_performance

    summary = json.loads((HERE / "card-performance/summary.json").read_text())
    if problem == "smoke":
        summary["selection"]["source_count"] = 10
    elif problem == "filter":
        summary["selection"]["excluded_taxonomy_count"] = 0
    elif problem == "split":
        summary["report_count"] -= 1
    elif problem == "taxonomy":
        summary["unmapped_truth"] = 1
    elif problem == "model":
        summary["models"].pop("nemo-tta")
    elif problem == "threshold":
        summary["models"]["nemo"].pop("threshold")
    if problem != "missing":
        (tmp_path / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError):
        card_performance(tmp_path, required=True)


def test_runtime_bootstrap_omits_binary_card_assets(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from dev.releases.mambo_v3 import package_download_metadata as packaging

    source = tmp_path / "source"
    source.mkdir()
    (source / "classes.json").write_text('{"labels": []}')
    manifest = {"files": {"classes.json": {}, "performance/quality.png": {}}, "origins": {}}
    monkeypatch.setattr(packaging, "Bundle", lambda root: SimpleNamespace(manifest=manifest, file=lambda name: source / name))
    target = tmp_path / "bootstrap.json"
    packaging.package(source, target)
    metadata = json.loads(target.read_text())["metadata"]
    assert json.loads(metadata["classes.json"]) == {"labels": []}
    assert "performance/quality.png" not in json.loads(metadata["release.json"])["files"]
