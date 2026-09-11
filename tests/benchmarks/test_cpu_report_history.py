import hashlib
import json

import pytest

from dev.benchmarks.reporting.release_history import record_identity
from dev.benchmarks.reporting.report_history import METRICS, archive, render


@pytest.fixture
def cpu_report(tmp_path):
    source = {
        "schema_version": 1,
        "status": "evaluated",
        "phase": "complete",
        "required_candidate_ops": ["QGemm"],
        "stages": [],
        "execution": {},
        "pairs": [{"trial": 0, "candidate_over_baseline": {"warm_latency": 0.5, "resident_bytes": 0.5, "peak_resident_bytes": 0.5}}],
    }

    def save(name, data):
        path = tmp_path / name / "report.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
        return hashlib.sha256(path.read_bytes()).hexdigest()

    quality = {
        "manifest": {"sha256": "d" * 64},
        "levels": [{"name": "leaf", "samples": 4, "prediction_changes": 0, "candidate_minus_baseline": dict.fromkeys(METRICS, 0)}],
        "models": {role: {"metrics": {metric: {"0": 0.5} for metric in METRICS}} for role in ("baseline", "candidate")},
    }
    source["quality"] = {**quality, "report_sha256": save("quality", quality)}
    for role, size in (("baseline", 100), ("candidate", 50)):
        files = [{"path": "/private/model", "sha256": role[0] * 64, "bytes": size}]
        execution = [{"op": "QGemm" if role == "candidate" else "Gemm", "provider": "CPUExecutionProvider", "count": 1}]
        source["execution"][role] = execution
        placement = {"inputs": {"sha256": "e" * 64}, "models": [{"files": files, "execution": execution}]}
        child = {
            "status": "measured",
            "model_files": files,
            "inputs": {"path": "/private/inputs", "sha256": "e" * 64},
            "median_seconds": size / 1000,
            "versions": dict.fromkeys(("onnxruntime", "onnx", "numpy"), "test"),
            "environment": {"platform": "Linux fixture", "machine": "aarch64", "python": "3.13", "cpu_affinity": [0]},
            "runtime_build": "test",
            "settings": {"threads": 1, "inter_op_threads": 1, "warmup": 1, "repeats": 3, "optimization": "all"},
            "memory": {"after_measurement": {"resident_bytes": size, "peak_resident_bytes": size * 2}},
        }
        for name, data, status in [(f"placement-{role}", placement, "passed"), (f"trial-0-{role}", child, "measured")]:
            source["stages"].append({"name": name, "status": status, "report_sha256": save(name, data)})
    path = tmp_path / "report.json"
    path.write_text(json.dumps(source))
    return path, source


def test_cpu_record_preserves_scope_without_gpu_fields(cpu_report, tmp_path):
    path, _ = cpu_report
    history = tmp_path / "history"
    record = archive(path, history, "cpu-1", "a" * 40, "simulated ARM fixture", performance_valid=True)
    data = json.loads(record.read_text())
    assert data["kind"] == "onnx_cpu_deployment"
    assert data["comparison"]["recorded_trials"] == 1
    assert data["evidence"]["execution"]["candidate"][0]["op"] == "QGemm"
    assert data["evidence"]["models"]["baseline"][0]["bytes"] == 100
    assert b"/private" not in record.read_bytes()
    assert record_identity(record.name, record.read_bytes())[1] == "a" * 40
    page = render(history).read_text()
    assert "CPU architecture: aarch64" in page and "separate-process medians" in page
    assert "approximate peak" in page and "Baseline device MiB" not in page
    assert "Measured GPU" not in page
    assert archive(path, history, "cpu-1", "a" * 40, "simulated ARM fixture", performance_valid=True) == record


@pytest.mark.parametrize("name", ["quality", "placement-candidate", "trial-0-candidate"])
def test_changed_cpu_child_evidence_is_rejected(cpu_report, tmp_path, name):
    path, _ = cpu_report
    (tmp_path / name / "report.json").write_text("{}")
    with pytest.raises(ValueError, match="evidence changed"):
        archive(path, tmp_path / "history", "cpu", "a" * 40, "CPU")


@pytest.mark.parametrize("defect", ["ratio", "missing-placement", "trial-index"])
def test_inconsistent_cpu_summary_is_rejected(cpu_report, tmp_path, defect):
    path, data = cpu_report
    if defect == "ratio":
        data["pairs"][0]["candidate_over_baseline"]["warm_latency"] = 0.1
    elif defect == "missing-placement":
        data["execution"] = {}
    else:
        data["pairs"][0]["trial"] = 2
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        archive(path, tmp_path / "history", "cpu", "a" * 40, "CPU")


def test_failed_cpu_evaluation_never_presents_performance(cpu_report, tmp_path):
    path, data = cpu_report
    data.update(status="failed", phase="resources", pairs=[])
    path.write_text(json.dumps(data))
    history = tmp_path / "history"
    record = archive(path, history, "cpu", "a" * 40, "CPU", performance_valid=True)
    assert not json.loads(record.read_text())["performance_valid"]
    assert "Resource readings excluded" in render(history).read_text()


@pytest.mark.parametrize("status", ["evaluated", "failed"])
def test_requested_cpu_budget_survives_archival(cpu_report, tmp_path, status):
    path, data = cpu_report
    data["status"] = status
    data["settings"] = {"threads": 1, "trials": 1 if status == "evaluated" else 3, "warmup": 1, "repeats": 3}
    path.write_text(json.dumps(data))
    history = tmp_path / "history"
    record = json.loads(archive(path, history, "cpu", "a" * 40, "CPU").read_text())
    assert record["comparison"]["requested_settings"] == data["settings"]
    assert f"1 of {data['settings']['trials']} requested" in render(history).read_text()


@pytest.mark.parametrize("key,value", [("trials", 2), ("threads", 2), ("warmup", 2), ("repeats", 4), ("trials", True), ("trials", 0)])
def test_incomplete_or_inconsistent_requested_cpu_settings_are_rejected(cpu_report, tmp_path, key, value):
    path, data = cpu_report
    data["settings"] = {"threads": 1, "trials": 1, "warmup": 1, "repeats": 3}
    data["settings"][key] = value
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="CPU|cpu"):
        archive(path, tmp_path / "history", "cpu", "a" * 40, "CPU")


def test_older_cpu_report_does_not_invent_requested_budget(cpu_report, tmp_path):
    history = tmp_path / "history"
    record = json.loads(archive(cpu_report[0], history, "cpu", "a" * 40, "CPU").read_text())
    assert "requested_settings" not in record["comparison"]
    assert "requested count was not retained" in render(history).read_text()
