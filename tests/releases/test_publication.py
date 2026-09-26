"""Publication preserves reviewed payloads and pins actual Hub upload commits."""

import json
import sys
from types import SimpleNamespace

import pytest

from dev.releases.mambo_v3.prepare_candidate import digest
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
