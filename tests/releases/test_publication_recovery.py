"""Recovery may finish a qualified release but cannot rebuild or substitute its inputs."""

import json
from copy import deepcopy

import pytest

from dev.releases.mambo_v3.prepare_candidate import digest
from dev.releases.mambo_v3.recover_publication import validate_payload, validate_run


@pytest.fixture
def publication():
    return {
        "run": {"event": "release", "path": ".github/workflows/publish-model.yml", "status": "completed", "head_sha": "a" * 40},
        "jobs": [{"name": name, "conclusion": "success"} for name in ("prepare", "package")],
        "release": {"draft": False, "prerelease": False, "published_at": "2026-09-26T20:00:00Z"},
        "tag_commit": "a" * 40,
    }


def test_recover_qualified_published_run(publication):
    assert validate_run(**publication) == "a" * 40


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("run", "event", "workflow_dispatch"),
        ("run", "path", ".github/workflows/ci.yml"),
        ("run", "status", "in_progress"),
        ("run", "head_sha", "b" * 40),
        ("release", "draft", True),
        ("release", "prerelease", True),
    ],
)
def test_recovery_rejects_unpublished_or_different_source(publication, section, key, value):
    publication[section][key] = value
    with pytest.raises(ValueError):
        validate_run(**publication)


def test_recovery_requires_package_publication(publication):
    publication["jobs"][-1]["conclusion"] = "failure"
    with pytest.raises(ValueError, match="package publication"):
        validate_run(**publication)


def test_retained_artifact_identity_and_hashes(tmp_path):
    identity = {"source_commit": "a" * 40, "model_id": "MAMBO_v3", "package_version": "0.3.0"}
    candidate = {**identity, "distribution": "mambo-v3", "qualification": "passed"}
    for name in ("github", "model", "space"):
        folder = tmp_path / name
        folder.mkdir()
        (folder / "release-candidate.json").write_text(json.dumps(candidate))
        manifest = {**identity, "files": {"release-candidate.json": digest(folder / "release-candidate.json")}}
        (folder / "publication.json").write_text(json.dumps(manifest))
    validate_payload(tmp_path, "a" * 40, "mambo-v3", "0.3.0")
    with pytest.raises(ValueError, match="original release"):
        validate_payload(tmp_path, "b" * 40, "mambo-v3", "0.3.0")
    with pytest.raises(ValueError, match="qualified release"):
        validate_payload(tmp_path, "a" * 40, "another-model", "0.3.0")
    changed = deepcopy(candidate)
    changed["qualification"] = "failed"
    (tmp_path / "github/release-candidate.json").write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="integrity"):
        validate_payload(tmp_path, "a" * 40, "mambo-v3", "0.3.0")
