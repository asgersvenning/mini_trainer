import json
import subprocess
from types import SimpleNamespace

import pytest

from dev.benchmarks.reporting.release_history import GitHub, synchronize
from dev.benchmarks.reporting.report_history import archive


@pytest.fixture
def remote(monkeypatch):
    state = SimpleNamespace(releases=[], assets={}, calls=[], corrupt=False, denied=False)
    monkeypatch.delenv("GH_HOST", raising=False)

    def run(command, *, input, capture_output, check):
        assert command[:4] == ["gh", "api", "--hostname", "github.com"]
        assert capture_output and check
        state.calls.append(command)
        if state.denied:
            raise subprocess.CalledProcessError(1, command, stderr=b"secret-token")
        endpoint = command[4]
        if "--paginate" in command:
            assert "--slurp" in command and endpoint.endswith("?per_page=100")
            if "/assets?" in endpoint:
                release_id = int(endpoint.split("/")[-2])
                data = [{"id": key, "name": value[1]} for key, value in state.assets.items() if value[0] == release_id]
            else:
                data = state.releases
            # Exercise multiple pages, including an empty first page.
            payload = json.dumps([[], data[:1], data[1:]]).encode()
        elif "--method" in command:
            assert "--input" in command and command[-1] == "-"
            if endpoint.startswith("https://uploads.github.com/"):
                release_id = int(endpoint.split("/")[-2])
                name = endpoint.split("?name=")[1]
                assert not any(v[:2] == (release_id, name) for v in state.assets.values())
                asset_id = len(state.assets) + 1
                state.assets[asset_id] = (release_id, name, input)
                payload = json.dumps({"id": asset_id}).encode()
            else:
                data = json.loads(input)
                assert data["draft"] and data["prerelease"] and data["make_latest"] == "false"
                assert data["tag_name"].startswith("benchmark-history-")
                data["id"] = len(state.releases) + 1
                state.releases.append(data)
                payload = json.dumps(data).encode()
        else:
            assert "Accept: application/octet-stream" in command
            payload = state.assets[int(endpoint.rsplit("/", 1)[1])][2]
            if state.corrupt:
                payload += b"changed"
        return SimpleNamespace(stdout=payload)

    monkeypatch.setattr(subprocess, "run", run)
    return state


@pytest.fixture
def records(tmp_path):
    report = tmp_path / "status.json"
    report.write_text('{"phase":"preflight","exit_code":19}')
    history = tmp_path / "incoming"
    archive(report, history, "run-1", "a" * 40, "Blair / GPU")
    archive(report, history, "run-2", "b" * 40, "Blair / GPU")
    return history / "records"


def test_upload_restore_and_retry_are_byte_identical(remote, records, tmp_path):
    synchronize("owner/repo", tmp_path / "first", records, upload=True)
    assert len(remote.releases) == 1 and len(remote.assets) == 2
    before = len(remote.calls)
    synchronize("owner/repo", tmp_path / "retry", records, upload=True)
    assert not any("--method" in call for call in remote.calls[before:])
    page = synchronize("owner/repo", tmp_path / "restored")
    assert "Quality results unavailable" in page.read_text()
    for record in records.glob("*.json"):
        assert (page.parent / "records" / record.name).read_bytes() == record.read_bytes()


def test_preview_does_not_write_remote(remote, records, tmp_path):
    page = synchronize("owner/repo", tmp_path / "preview", records)
    assert page.is_file() and not remote.assets and not remote.releases


def test_multiple_months_restore_published_archives_and_ignore_package_releases(remote, records, tmp_path):
    for name, month in [("run-1", "2026-08"), ("run-2", "2026-09")]:
        path = records / f"{name}.json"
        data = json.loads(path.read_text())
        data["recorded_at"] = f"{month}-01T00:00:00+00:00"
        path.write_text(json.dumps(data))
    remote.releases.append({"id": 1, "tag_name": "v1.0", "draft": False})
    synchronize("owner/repo", tmp_path / "first", records, upload=True)
    assert [release["tag_name"] for release in remote.releases] == ["v1.0", "benchmark-history-2026-08", "benchmark-history-2026-09"]
    remote.releases[1]["draft"] = False
    remote.releases[1]["immutable"] = True
    page = synchronize("owner/repo", tmp_path / "restored", records, upload=True)
    assert "run-1" in page.read_text() and "run-2" in page.read_text()


def test_interrupted_upload_can_resume_without_deleting_assets(remote, records, tmp_path):
    remote.corrupt = True
    with pytest.raises(ValueError, match="readback"):
        synchronize("owner/repo", tmp_path / "interrupted", records, upload=True)
    first_asset = remote.assets[1]
    remote.corrupt = False
    synchronize("owner/repo", tmp_path / "retry", records, upload=True)
    assert remote.assets[1] == first_asset and len(remote.assets) == 2


def test_conflicting_identity_never_overwrites_remote(remote, records, tmp_path):
    synchronize("owner/repo", tmp_path / "first", records, upload=True)
    before = dict(remote.assets)
    record = records / "run-1.json"
    data = json.loads(record.read_text())
    data["note"] = "changed"
    record.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="different bytes"):
        synchronize("owner/repo", tmp_path / "conflict", records, upload=True)
    assert remote.assets == before


def test_api_failure_is_not_treated_as_empty_storage(remote, records, tmp_path):
    remote.denied = True
    with pytest.raises(RuntimeError, match="GitHub request failed") as error:
        synchronize("owner/repo", tmp_path / "denied", records, upload=True)
    assert "secret-token" not in str(error.value)
    assert len(remote.calls) == 1 and not remote.releases


def test_upload_requires_successful_readback(remote, records, tmp_path):
    remote.corrupt = True
    with pytest.raises(ValueError, match="readback"):
        synchronize("owner/repo", tmp_path / "corrupt", records, upload=True)
    assert len(remote.assets) == 1  # Preserve the uploaded bytes for inspection/retry.


def test_published_month_cannot_be_appended(remote, records, tmp_path):
    synchronize("owner/repo", tmp_path / "first", records, upload=True)
    remote.releases[0]["draft"] = False
    path = records / "run-1.json"
    data = json.loads(path.read_text())
    data["run_id"] = "run-3"
    (records / "run-3.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="published"):
        synchronize("owner/repo", tmp_path / "published", records, upload=True)
    assert len(remote.assets) == 2


def test_malformed_record_is_rejected_before_remote_writes(remote, records, tmp_path):
    path = records / "run-1.json"
    data = json.loads(path.read_text())
    del data["evidence"]
    path.write_text(json.dumps(data))
    with pytest.raises(KeyError):
        synchronize("owner/repo", tmp_path / "invalid", records, upload=True)
    assert not remote.assets and not remote.releases


def test_remote_traversal_cannot_escape_output(remote, records, tmp_path):
    synchronize("owner/repo", tmp_path / "first", records, upload=True)
    release, _, payload = remote.assets[1]
    remote.assets[1] = (release, "../escape.json", payload)
    with pytest.raises(ValueError, match="filename"):
        synchronize("owner/repo", tmp_path / "invalid")
    assert not (tmp_path / "invalid/escape.json").exists()


def test_missing_cli_and_unsupported_host_have_actionable_errors(monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError

    monkeypatch.delenv("GH_HOST", raising=False)
    monkeypatch.setattr(subprocess, "run", missing)
    with pytest.raises(RuntimeError, match="Install GitHub CLI"):
        GitHub("owner/repo").listing("repos/owner/repo/releases")
    monkeypatch.setenv("GH_HOST", "other.example")
    with pytest.raises(ValueError, match="github.com only"):
        GitHub("owner/repo")


@pytest.mark.parametrize("repository", ["../repo", "owner/../repo", "owner/repo?token=bad", "-owner"])
def test_repository_names_are_restricted(repository):
    with pytest.raises(ValueError):
        GitHub(repository)
