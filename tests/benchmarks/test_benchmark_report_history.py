import hashlib
import json

import pytest

from dev.benchmarks.report_history import METRICS, archive, render


@pytest.fixture
def report(tmp_path):
    child = tmp_path / "latency-0"
    child.mkdir()
    (child / "report.json").write_text(
        json.dumps(
            {
                "versions": {"torch": "test", "tensorrt": "test", "numpy": "test", "private": "/private/runtime"},
                "environment": {"gpu": "GPU", "compute_capability": [8, 6], "platform": "Linux", "python": "3.13"},
            }
        )
    )
    snapshots = {
        name: {"device_used_bytes": value, "host": {"resident_bytes": value}} for name, value in [("cuda_initialized", 100), ("warm", 200)]
    }
    data = {
        "schema_version": 1,
        "status": "evaluated",
        "phase": "complete",
        "settings": {"trials": 1},
        "builds": {
            role: {"directory": "/private/model", "engine": {"sha256": "a" * 64, "bytes": 1, "context_memory_bytes": 2}}
            for role in ("baseline", "candidate")
        },
        "quality": {
            "levels": [{"name": "leaf", "samples": 10, "prediction_changes": 2, "candidate_minus_baseline": dict.fromkeys(METRICS, 0.1)}],
            "models": {
                role: {"source_path": "/private/predictions.csv", "metrics": {key: {"0": value} for key in METRICS}}
                for role, value in [("baseline", 0.5), ("candidate", 0.6)]
            },
        },
        "trials": [
            {
                "trial": 0,
                "latency": {"median_paired_ratio": 0.5, "median_seconds": {"baseline": 0.2, "candidate": 0.1}},
                "memory": {role: snapshots for role in ("baseline", "candidate")},
            }
        ],
        "stages": [
            {"name": "latency-0", "status": "passed", "report_sha256": hashlib.sha256((child / "report.json").read_bytes()).hexdigest()}
        ],
    }
    path = tmp_path / "report.json"
    path.write_text(json.dumps(data))
    return path, data


def test_archive_is_idempotent_but_rejects_changed_identity(report, tmp_path):
    path, _ = report
    history = tmp_path / "history"
    archived = archive(path, history, "run-1", "a" * 40, "profile")
    before = archived.read_bytes()
    assert archive(path, history, "run-1", "a" * 40, "profile").read_bytes() == before
    with pytest.raises(ValueError, match="already exists"):
        archive(path, history, "run-1", "b" * 40, "profile")
    assert archived.read_bytes() == before and not list((history / "records").glob("*.tmp"))
    assert b"/private" not in before
    document = render(history).read_text()
    assert "Resource readings excluded" in document and "Paired latency ratio" not in document
    assert "Measured GPU: GPU" in document


@pytest.mark.parametrize("kind", ["delta", "nonfinite", "missing-trial"])
def test_invalid_evidence_is_not_archived(report, tmp_path, kind):
    path, data = report
    if kind == "delta":
        data["quality"]["levels"][0]["candidate_minus_baseline"]["f1"] = 0.9
    if kind == "nonfinite":
        data["quality"]["models"]["baseline"]["metrics"]["f1"]["0"] = float("nan")
    if kind == "missing-trial":
        data["trials"] = []
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        archive(path, tmp_path / "history", "run", "a" * 40, "profile")
    assert not (tmp_path / "history").exists()


def test_hash_changed_runtime_evidence_is_rejected(report, tmp_path):
    path, _ = report
    (tmp_path / "latency-0/report.json").write_text("{}")
    with pytest.raises(ValueError, match="evidence changed"):
        archive(path, tmp_path / "history", "run", "a" * 40, "profile")


def test_undefined_metrics_and_escaped_labels_remain_visible(report, tmp_path):
    path, data = report
    data["quality"]["models"]["baseline"]["metrics"]["f1"]["0"] = None
    data["quality"]["levels"][0]["candidate_minus_baseline"]["f1"] = None
    path.write_text(json.dumps(data))
    history = tmp_path / "history"
    archive(path, history, "run", "a" * 40, "<script>alert(1)</script>", performance_valid=True, note="<img src=x onerror=alert(1)>")
    document = render(history).read_text()
    assert "undefined" in document and "Paired latency ratio" in document
    assert "<script>" not in document and "<img " not in document
    assert "&lt;script&gt;" in document


def test_early_failure_is_visible_without_performance_claim(tmp_path):
    path = tmp_path / "status.json"
    path.write_text(json.dumps({"phase": "preflight", "exit_code": 19}))
    history = tmp_path / "history"
    record = archive(path, history, "failed", "a" * 40, "profile", performance_valid=True)
    assert not json.loads(record.read_text())["performance_valid"]
    document = render(history).read_text()
    assert "failed" in document and "Quality results unavailable" in document and "Resource readings excluded" in document
    path.write_text(json.dumps({"phase": "complete", "exit_code": 0}))
    with pytest.raises(ValueError, match="requires its evaluation report"):
        archive(path, history, "new", "a" * 40, "profile")


@pytest.mark.parametrize("run_id,url", [("../escape", None), ("safe", "javascript:alert(1)"), ("safe", "https://user:pass@example.com")])
def test_unsafe_names_and_links_are_rejected(report, tmp_path, run_id, url):
    with pytest.raises(ValueError):
        archive(report[0], tmp_path / "history", run_id, "a" * 40, "profile", run_url=url)
