"""Exercise the standalone calibrator without mini_trainer or remote storage."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

SCRIPT = Path(__file__).parents[2] / "dev/ucloud/calibrate_io.py"
spec = importlib.util.spec_from_file_location("calibrate_io", SCRIPT)
calibrator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(calibrator)


@pytest.mark.parametrize("mode", ["read", "stage", "decode"])
def test_standalone_trials_are_disjoint_and_leave_sources_intact(tmp_path, mode):
    source = tmp_path / "images"
    source.mkdir()
    image = Image.new("RGB", (4, 4), "red")
    for index in range(128):
        image.save(source / f"{index}.png")
    before = {path.name: path.read_bytes() for path in source.iterdir()}
    destination = tmp_path / "destination"
    destination.mkdir()
    output = tmp_path / "report.json"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(source),
            "--mode",
            mode,
            "--destination",
            str(destination),
            "--output",
            str(output),
            "--workers",
            "1,4",
            "--files-per-trial",
            "32",
            "--confirmation-rounds",
            "1",
            "--budget-seconds",
            "30",
            "--trial-seconds",
            "5",
            "--resize",
            "2",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=40,
    )
    report = json.loads(output.read_text())
    paths = [path for selection in report["selection_paths"] for path in selection]
    assert len(paths) == len(set(paths)) == 128
    assert len(report["trials"]) == 4
    assert all(row["completed"] == 32 and row["eligible"] for row in report["trials"])
    assert report["recommendation"]["workers"] in (1, 4)
    assert {path.name: path.read_bytes() for path in source.iterdir()} == before
    assert list(destination.iterdir()) == []


def test_recommendation_prefers_near_best_and_rejects_failed_setting():
    rows = [
        {"workers": workers, "phase": "confirm", "images_per_second": rate, "eligible": eligible}
        for workers, rate, eligible in [(32, 98, True), (128, 100, True), (512, 200, False)]
    ]
    assert calibrator.recommend(rows, 0.05)["workers"] == 32
    assert calibrator.recommend(rows, 0)["workers"] == 128
    assert calibrator.recommend([], 0.05)["workers"] is None


def test_blocked_read_is_terminated_without_hanging_calibration(tmp_path):
    import os
    import time
    from types import SimpleNamespace

    source = tmp_path / "blocked.jpg"
    os.mkfifo(source)
    args = SimpleNamespace(mode="read", resize=0, max_mib=1, trial_seconds=0.3, max_rss_mib=4096)
    row = calibrator.run_trial(args, [str(source)], 1, "sweep", tmp_path, time.monotonic() + 5)
    assert row["termination"] == "time_limit"
    assert row["completed"] == 0
    assert not row["eligible"]


def test_timeout_does_not_consume_unsubmitted_paths(tmp_path):
    import os
    import time
    from types import SimpleNamespace

    paths = []
    for index in range(100):
        path = tmp_path / f"{index}.jpg"
        os.mkfifo(path)
        paths.append(str(path))
    args = SimpleNamespace(mode="read", resize=0, max_mib=1, trial_seconds=0.5, max_rss_mib=4096)
    row = calibrator.run_trial(args, paths, 1, "sweep", tmp_path, time.monotonic() + 5)
    assert row["selected"] == 100
    assert row["attempted"] == 1
    assert row["completed"] == 0


def test_byte_limit_shrinks_trial_instead_of_aborting(tmp_path):
    import time
    from types import SimpleNamespace

    paths = []
    for index in range(3):
        path = tmp_path / f"{index}.jpg"
        path.write_bytes(b"x" * 600000)
        paths.append(str(path))
    args = SimpleNamespace(mode="read", resize=0, max_mib=1, trial_seconds=5, max_rss_mib=4096)
    row = calibrator.run_trial(args, paths, 512, "sweep", tmp_path, time.monotonic() + 10)
    assert row["completed"] == row["selected"] == row["attempted"] == 1
    assert row["workers"] == 1
    assert row["requested_workers"] == 512
    assert row["eligible"]


def test_ten_file_trials_and_existing_output_work(tmp_path):
    source = tmp_path / "images"
    source.mkdir()
    for index in range(40):
        (source / f"{index}.jpg").write_bytes(b"encoded bytes")
    output = tmp_path / "report.json"
    output.write_text("previous report")
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(source),
            "--mode",
            "read",
            "--output",
            str(output),
            "--workers",
            "1,4,16",
            "--files-per-trial",
            "10",
            "--confirmation-rounds",
            "0",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert output.read_text() == "previous report"
    report = json.loads((tmp_path / "report-2.json").read_text())
    assert len(report["trials"]) == 3
    assert all(row["completed"] == 10 for row in report["trials"])
    assert report["recommendation"]["workers"] is not None
    assert "report-2.json" in result.stdout


def test_startup_time_is_separate_from_io_time(tmp_path, monkeypatch):
    import time

    source = tmp_path / "image.jpg"
    source.write_bytes(b"image bytes")
    job = {
        "mode": "read",
        "paths": [str(source)],
        "workers": 4,
        "resize": 0,
        "result": str(tmp_path / "result.json"),
        "ready": str(tmp_path / "ready.json"),
        "attempted": str(tmp_path / "attempted.bin"),
    }
    job_path = tmp_path / "job.json"
    job_path.write_text(json.dumps(job))
    original_clock = time.monotonic
    offset = [0]
    monkeypatch.setattr(calibrator.time, "monotonic", lambda: original_clock() + offset[0])
    original_warm = calibrator.warm_pool

    def delayed_warm(pool, workers):
        original_warm(pool, workers)
        offset[0] += 100  # Simulate slow initialization without a wall-clock sleep.

    monkeypatch.setattr(calibrator, "warm_pool", delayed_warm)
    calibrator.child(job_path)
    ready = json.loads((tmp_path / "ready.json").read_text())
    result = json.loads((tmp_path / "result.json").read_text())
    assert ready["initialization_seconds"] >= 100
    assert result["seconds"] < 10
    assert result["completed"] == 1


def test_failed_confirmation_keeps_provisional_sweep():
    rows = [
        {"workers": 64, "phase": "sweep", "images_per_second": 100, "eligible": True},
        {"workers": 64, "phase": "confirm", "images_per_second": 0, "eligible": False},
    ]
    result = calibrator.recommend(rows, 0.05)
    assert result["workers"] == 64
    assert result["basis"] == "provisional_sweep"
    assert "No usable confirmation" in result["confirmation_note"]


def test_summary_omits_large_manifest_and_keeps_error():
    text = calibrator.summary({"selection_paths": [["secretly-long-path"] * 10000], "error": "Unable to terminate process 42"})
    assert "secretly-long-path" not in text
    assert "Unable to terminate process 42" in text
    assert len(text.splitlines()) < 5


def test_memory_failure_is_not_recommended_from_sweep():
    rows = [
        {"workers": 64, "phase": "sweep", "images_per_second": 100, "eligible": True},
        {"workers": 64, "phase": "confirm", "images_per_second": 0, "eligible": False, "termination": "memory_limit"},
    ]
    assert calibrator.recommend(rows, 0.05)["workers"] is None


def test_termination_failure_is_saved_without_terminal_traceback(tmp_path, monkeypatch):
    source = tmp_path / "images"
    source.mkdir()
    (source / "image.jpg").write_bytes(b"image")
    output = tmp_path / "report.json"
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), str(source), "--output", str(output), "--workers", "1"])

    def failure(*args, **kwargs):
        raise RuntimeError("I/O process 123 remains blocked after termination")

    monkeypatch.setattr(calibrator, "run_trial", failure)
    with pytest.raises(SystemExit, match="report.error.log"):
        calibrator.main()
    report = json.loads(output.read_text())
    assert "process 123" in report["error"]
    assert "process 123" in output.with_suffix(".summary.txt").read_text()
    assert "RuntimeError" in output.with_suffix(".error.log").read_text()
