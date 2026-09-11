import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from dev.benchmarks.inference import tensorrt_deployment as deployment
from dev.benchmarks.inference.onnx_inference import file_hash


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    builds = []
    for role in ("baseline", "candidate"):
        path = tmp_path / role
        path.mkdir()
        (path / "model.engine").write_bytes(role.encode())
        (path / "layers.json").write_text(json.dumps({"Layers": []}))
        (path / "report.json").write_text(
            json.dumps(
                {
                    "status": "passed",
                    "engine": {"sha256": file_hash(path / "model.engine"), "layers_sha256": file_hash(path / "layers.json")},
                    "settings": {},
                }
            )
        )
        builds.append(path)
    manifest, inputs = tmp_path / "manifest.json", tmp_path / "inputs.npz"
    manifest.write_text("{}")
    inputs.write_bytes(b"input payload")
    versions = {"tensorrt": "test", "torch": "test", "numpy": "test"}
    runtime = {"tensorrt": "test", "torch": "test", "gpu": "test"}
    calls = []

    def quality(manifest, baseline, candidate, output, *args):
        output.mkdir()
        result = {"levels": [], "models": {}, "undefined_metrics": ["retained undefined metric"]}
        (output / "report.json").write_text(json.dumps(result))
        (output / "summary.md").write_text("undefined metric remains visible")
        for role, engine in [("baseline", baseline), ("candidate", candidate)]:
            (output / role).mkdir()
            (output / role / "report.json").write_text(
                json.dumps(
                    {
                        "model_files": [{"sha256": file_hash(engine)}],
                        "manifest": {"sha256": file_hash(manifest)},
                        "batches": [{"sha256": "same input"}],
                        "runtime": runtime,
                    }
                )
            )
        return result

    def child(command, **kwargs):
        calls.append(command)
        out = Path(command[command.index("--output") + 1])
        out.mkdir()
        result = {
            "status": "passed",
            "inputs": {"sha256": file_hash(inputs)},
            "versions": versions,
            "environment": {"gpu": "test"},
            "settings": {},
        }
        if command[2].endswith("tensorrt_pair"):
            result.update(
                models={role: {"sha256": file_hash(path / "model.engine")} for role, path in zip(("baseline", "candidate"), builds)},
                summary={"median_paired_ratio": 1.0},
            )
        else:
            engine = Path(command[command.index("--engine") + 1])
            result.update(engine={"sha256": file_hash(engine)}, memory={"warm": {"device_used_bytes": 100, "host": {"resident_bytes": 50}}})
        (out / "report.json").write_text(json.dumps(result))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(deployment, "run_pair", quality)
    monkeypatch.setattr(deployment.subprocess, "run", child)
    return builds, manifest, inputs, calls, quality, child


def test_pipeline_alternates_memory_and_retains_quality(pipeline, tmp_path):
    builds, manifest, inputs, calls, _, _ = pipeline
    output = tmp_path / "result"
    result = deployment.evaluate(*builds, manifest, inputs, output, trials=2)
    assert result["status"] == "evaluated"
    assert result["quality"]["undefined_metrics"]
    assert "undefined" in (output / "summary.md").read_text()
    assert [stage["name"] for stage in result["stages"]] == [
        "latency-0",
        "memory-0-baseline",
        "memory-0-candidate",
        "latency-1",
        "memory-1-candidate",
        "memory-1-baseline",
    ]
    assert "--reverse" not in calls[0] and "--reverse" in calls[3]
    with pytest.raises(FileExistsError):
        deployment.evaluate(*builds, manifest, inputs, output)


@pytest.mark.parametrize("artifact", ["model.engine", "layers.json"])
def test_mismatched_build_artifacts_fail_before_runtime(pipeline, tmp_path, artifact):
    builds, manifest, inputs, calls, _, _ = pipeline
    (builds[1] / artifact).write_bytes(b"changed")
    with pytest.raises(ValueError):
        deployment.evaluate(*builds, manifest, inputs, tmp_path / "result")
    report = json.loads((tmp_path / "result/report.json").read_text())
    assert report["status"] == "failed" and report["phase"] == "inspection" and not calls


@pytest.mark.parametrize("mutation", ["input", "engine", "process"])
def test_failed_or_changed_resource_stops_pipeline(pipeline, tmp_path, monkeypatch, mutation):
    builds, manifest, inputs, calls, _, child = pipeline

    def change(command, **kwargs):
        result = child(command, **kwargs)
        path = Path(command[command.index("--output") + 1]) / "report.json"
        report = json.loads(path.read_text())
        if mutation == "input":
            report["inputs"]["sha256"] = "changed"
        elif mutation == "engine":
            report["models"]["candidate"]["sha256"] = "changed"
        else:
            return SimpleNamespace(returncode=17)
        path.write_text(json.dumps(report))
        return result

    monkeypatch.setattr(deployment.subprocess, "run", change)
    with pytest.raises((ValueError, RuntimeError)):
        deployment.evaluate(*builds, manifest, inputs, tmp_path / "result")
    result = json.loads((tmp_path / "result/report.json").read_text())
    assert result["status"] == "failed" and result["phase"] == "resources"
    assert len(calls) == 1 and (tmp_path / "result/quality/report.json").exists()


def test_changed_quality_batches_stop_before_resources(pipeline, tmp_path, monkeypatch):
    builds, manifest, inputs, calls, quality, _ = pipeline

    def changed(*args):
        result = quality(*args)
        path = args[3] / "candidate/report.json"
        data = json.loads(path.read_text())
        data["batches"][0]["sha256"] = "changed"
        path.write_text(json.dumps(data))
        return result

    monkeypatch.setattr(deployment, "run_pair", changed)
    with pytest.raises(ValueError, match="input batches differ"):
        deployment.evaluate(*builds, manifest, inputs, tmp_path / "result")
    assert not calls
    assert json.loads((tmp_path / "result/report.json").read_text())["status"] == "failed"


def test_help_is_runtime_independent():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import runpy, sys
sys.argv = ['tensorrt_deployment', '--help']
try:
    runpy.run_module('dev.benchmarks.inference.tensorrt_deployment', run_name='__main__')
except SystemExit as error:
    assert error.code == 0
assert 'torch' not in sys.modules and 'tensorrt' not in sys.modules
""",
        ],
        check=True,
        capture_output=True,
    )
