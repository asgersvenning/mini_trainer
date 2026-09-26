"""Publication must reject changed bytes and must not repeat immutable Hub uploads."""

import json
import sys
from types import SimpleNamespace

import pytest

from dev.releases.mambo_v3.prepare_candidate import digest
from dev.releases.mambo_v3.publish_assets import hub, verified_payload


@pytest.fixture
def payload(tmp_path):
    (tmp_path / "README.md").write_text("Public model card")
    data = {
        "source_commit": "a" * 40,
        "package_version": "0.3.0",
        "model_id": "MAMBO_v3",
        "files": {"README.md": digest(tmp_path / "README.md")},
    }
    (tmp_path / "publication.json").write_text(json.dumps(data))
    return tmp_path


def test_modified_or_uninventoried_file_cannot_be_published(payload):
    verified_payload(payload)
    (payload / "README.md").write_text("Unreviewed change")
    with pytest.raises(ValueError, match="integrity"):
        verified_payload(payload)
    (payload / "private.txt").write_text("Not a release input")
    with pytest.raises(ValueError, match="file set"):
        verified_payload(payload)


def test_hub_retry_checks_identity_without_reupload(payload, monkeypatch, tmp_path):
    remote = tmp_path / "remote.json"
    remote.write_bytes((payload / "publication.json").read_bytes())
    # Keep simulated remote state outside the local publication inventory.
    remote_state = remote.read_text()
    remote.unlink()
    calls = []
    api = SimpleNamespace(
        token="unused",
        list_repo_refs=lambda *a, **kw: SimpleNamespace(tags=[SimpleNamespace(name="v0.3.0")]),
        upload_folder=lambda **kw: calls.append(kw),
        create_tag=lambda **kw: calls.append(kw),
    )

    def download(*a, **kw):
        destination = tmp_path.parent / f"{tmp_path.name}-remote.json"
        destination.write_text(remote_state)
        return str(destination)

    monkeypatch.setenv("HF_TOKEN", "unused")
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=lambda **kw: api, hf_hub_download=download))
    hub(payload, "owner/model", "model")
    assert calls == []
    remote_state = json.dumps({"different": "release"})
    with pytest.raises(ValueError, match="Refusing to replace"):
        hub(payload, "owner/model", "model")
    assert calls == []


def test_hub_tags_exact_uploaded_commit(payload, monkeypatch):
    calls = []
    api = SimpleNamespace(
        token="unused",
        list_repo_refs=lambda *a, **kw: SimpleNamespace(tags=[]),
        upload_folder=lambda **kw: SimpleNamespace(oid="reviewed-upload-commit"),
        create_tag=lambda **kw: calls.append(kw),
    )
    monkeypatch.setenv("HF_TOKEN", "unused")
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=lambda **kw: api, hf_hub_download=None))
    hub(payload, "owner/model", "model")
    assert calls[0]["revision"] == "reviewed-upload-commit"
    assert calls[0]["tag"] == "v0.3.0"
