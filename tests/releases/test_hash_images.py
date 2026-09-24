"""Concurrent cold reads retain serial hashes and resumable input identity."""

import hashlib
import threading

import pytest

from dev.releases.mambo_v3 import hash_images as module


def inputs(root, count=12):
    records = []
    for i in range(count):
        path = root / f"{i}.jpg"
        path.write_bytes(f"image {i}".encode())
        records.append({"path": path.name, "labels": ["species", "genus", "family"]})
    return records


def test_concurrent_reads_preserve_hashes_order_and_resume(tmp_path, monkeypatch):
    records = inputs(tmp_path)
    original = module.file_hash
    barrier = threading.Barrier(4)
    lock = threading.Lock()
    calls = []

    def read(path):
        with lock:
            calls.append(path)
            first = len(calls) <= 4
        if first:
            barrier.wait(timeout=5)  # A serial implementation cannot complete this.
        return original(path)

    monkeypatch.setattr(module, "file_hash", read)
    checkpoint = tmp_path / "hashes.sqlite3"
    module.hash_images(records, tmp_path, checkpoint, {"snapshot": "original"}, workers=4)
    assert len(calls) == 12
    assert [r["sha256"] for r in records] == [hashlib.sha256(f"image {i}".encode()).hexdigest() for i in range(12)]
    fresh = [{k: v for k, v in r.items() if k != "sha256"} for r in records]
    monkeypatch.setattr(module, "file_hash", lambda path: pytest.fail("Unchanged cached file was reread"))
    module.hash_images(fresh, tmp_path, checkpoint, {"snapshot": "original"}, workers=4)
    assert fresh == records
    (tmp_path / "0.jpg").write_bytes(b"changed image bytes")
    monkeypatch.setattr(module, "file_hash", original)
    fresh = [{k: v for k, v in r.items() if k != "sha256"} for r in records]
    module.hash_images(fresh, tmp_path, checkpoint, {"snapshot": "original"}, workers=4)
    assert fresh[0]["sha256"] == original(tmp_path / "0.jpg")
    with pytest.raises(ValueError, match="different inputs"):
        module.hash_images([], tmp_path, checkpoint, {"snapshot": "other"}, workers=4)


def test_failure_saves_completed_hashes_and_rejects_unsafe_paths(tmp_path):
    records = inputs(tmp_path)
    (tmp_path / "5.jpg").unlink()
    checkpoint = tmp_path / "hashes.sqlite3"
    with pytest.raises(FileNotFoundError):
        module.hash_images(records, tmp_path, checkpoint, {}, workers=1)
    import sqlite3

    with sqlite3.connect(checkpoint) as connection:
        assert connection.execute("SELECT COUNT(*) FROM hashes").fetchone()[0] > 0
    with pytest.raises(ValueError, match="Unsafe"):
        module.hash_images([{"path": "../escape.jpg"}], tmp_path, tmp_path / "unsafe.sqlite3", {}, workers=1)


def test_default_readers_use_allocated_cpu_affinity(monkeypatch):
    monkeypatch.setattr(module.os, "sched_getaffinity", lambda pid: set(range(48)))
    assert module.default_workers() == 96
