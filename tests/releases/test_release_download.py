"""Verified automatic model caching, offline behavior and local-bundle isolation."""

import hashlib
import io
import json
from pathlib import Path

import pytest

from deployment.mambo_deploy import download
from deployment.mambo_deploy.bundle import Bundle


def test_download_publishes_only_verified_bytes_and_reuses_cache(tmp_path, monkeypatch):
    payload = b"model bytes"
    calls = []

    def get(url, timeout):
        calls.append(url)
        return io.BytesIO(payload)

    monkeypatch.setattr(download, "urlopen", get)
    target = tmp_path / "weights"
    args = dict(size=len(payload), sha256=hashlib.sha256(payload).hexdigest())
    download.fetch_file("https://example.test/weights", target, **args)
    download.fetch_file("https://example.test/weights", target, **args, offline=True)
    assert target.read_bytes() == payload and len(calls) == 1
    target.write_bytes(b"changed")
    with pytest.raises(ValueError, match="integrity"):
        download.fetch_file("https://example.test/weights", target, **args)
    assert len(calls) == 1


@pytest.mark.parametrize("payload", [b"short", b"bad-data", b"too-long-data"])
def test_failed_download_never_leaves_a_completed_file(tmp_path, monkeypatch, payload):
    monkeypatch.setattr(download, "urlopen", lambda *a, **k: io.BytesIO(payload))
    with pytest.raises(ValueError):
        download.fetch_file("https://example.test/weights", tmp_path / "model", size=8, sha256=hashlib.sha256(b"expected").hexdigest())
    assert list(tmp_path.iterdir()) == []


def test_offline_missing_cache_does_not_connect(tmp_path, monkeypatch):
    monkeypatch.setattr(download, "urlopen", lambda *a, **k: pytest.fail("network used"))
    with pytest.raises(FileNotFoundError, match="not cached"):
        download.fetch_file("https://example.test/weights", tmp_path / "model", size=0, sha256="0" * 64, offline=True)


def test_packaged_metadata_and_automatic_predictor(tmp_path, monkeypatch):
    from deployment.mambo_deploy import Predictor

    monkeypatch.delenv("MAMBO_BUNDLE", raising=False)
    monkeypatch.delenv("MAMBO_OFFLINE", raising=False)
    monkeypatch.setenv("MAMBO_CACHE", str(tmp_path))
    monkeypatch.setattr(download, "urlopen", lambda *a, **k: pytest.fail("metadata should be packaged"))
    p = Predictor()
    assert p.bundle.download and p.preset == "full"
    monkeypatch.setenv("MAMBO_OFFLINE", "1")
    assert Predictor().bundle.root == p.bundle.root
    with pytest.raises(FileNotFoundError, match="not cached"):
        p.bundle.profile("onnx")
    explicit = Predictor(bundle=p.bundle.root)
    assert not explicit.bundle.download
    with pytest.raises(FileNotFoundError):
        explicit.bundle.profile("onnx")
    manifest = json.loads((p.bundle.root / "release.json").read_text())
    for relative in manifest["files"]:
        if relative not in manifest["origins"]:
            Bundle(p.bundle.root).file(relative)
    assert Path(download.__file__).with_name("default_bundle.json").exists()


@pytest.mark.parametrize("hardlink", [True, False])
def test_weight_bytes_survive_metadata_revision_offline(tmp_path, monkeypatch, hardlink):
    payload = b"same immutable model"
    calls = []

    def get(*args, **kwargs):
        calls.append(True)
        return io.BytesIO(payload)

    monkeypatch.setattr(download, "urlopen", get)
    if not hardlink:

        def deny_link(*args):
            raise OSError("cross-filesystem")

        monkeypatch.setattr(download.os, "link", deny_link)
    args = dict(cache=tmp_path, size=len(payload), sha256=hashlib.sha256(payload).hexdigest())
    first = tmp_path / "revision-1/models/model.onnx.data"
    second = tmp_path / "revision-2/models/model.onnx.data"
    download.cached_model_file("https://example.test/model", first, **args)
    download.cached_model_file("https://example.test/model", second, **args, offline=True)
    assert first.read_bytes() == second.read_bytes() == payload
    assert len(calls) == 1
    assert not first.is_symlink() and not second.is_symlink()
    assert not list(second.parent.glob(".model-*"))


def test_offline_can_materialize_packaged_metadata_without_connecting(tmp_path, monkeypatch):
    from deployment.mambo_deploy import Predictor

    monkeypatch.delenv("MAMBO_BUNDLE", raising=False)
    monkeypatch.setenv("MAMBO_CACHE", str(tmp_path))
    monkeypatch.setenv("MAMBO_OFFLINE", "1")
    monkeypatch.setattr(download, "urlopen", lambda *args, **kwargs: pytest.fail("network used"))
    predictor = Predictor()
    assert predictor.preset == "full"
    with pytest.raises(FileNotFoundError, match="not cached"):
        predictor.bundle.profile("onnx")
