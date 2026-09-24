"""Monitoring existing logs must not change evidence or invent throughput."""

import json
import os

import pytest

from dev.monitor_mambo_release import Rate, checkpoint, snapshot


@pytest.mark.parametrize("line", ["1600 632913\n", "torch cuda:0: 1600/632913\n", "onnx cpu: 1600/632913\n"])
def test_reads_existing_formats(tmp_path, line):
    path = tmp_path / "job.log"
    path.write_text("warning from runtime\n" + line + "another warning\n")
    assert checkpoint(path)[:2] == (1600, 632913)
    assert path.read_text().endswith("another warning\n")


def test_rate_waits_for_progress_and_rejects_stale_or_restarted_logs():
    rate = Rate()
    assert rate.estimate(100, 1000, 100, 100) is None
    assert rate.estimate(100, 1000, 100, 105) is None
    assert rate.estimate(200, 1000, 110, 110) == (10, 80)
    assert rate.estimate(200, 1000, 110, 200) is None
    assert rate.estimate(32, 1000, 210, 210) is None


def test_snapshot_distinguishes_completed_active_queued_and_failed(tmp_path):
    plan = {"status": "running", "completed": ["v2"], "jobs": [{"name": n} for n in ("v2", "torch", "onnx")]}
    p = tmp_path / "plan.json"
    p.write_text(json.dumps(plan))
    log = tmp_path / "torch.log"
    log.write_text("torch cuda:0: 32/632913\n")
    os.utime(log, (100, 100))
    original = p.read_bytes(), log.read_bytes()
    rates = {}
    rows, finished = snapshot(tmp_path, rates, 100)
    assert not finished
    assert "v2: complete" in rows and "onnx: queued" in rows
    assert "ETA unavailable" in rows[2]
    log.write_text("torch cuda:0: 32/632913\ntorch cuda:0: 1632/632913\n")
    os.utime(log, (110, 110))
    rows, _ = snapshot(tmp_path, rates, 110)
    assert "160.0 images/s" in rows[2]
    assert p.read_bytes() == original[0]
    (tmp_path / "torch").mkdir()
    (tmp_path / "torch/report.json").write_text('{"status": "failed"}')
    rows, _ = snapshot(tmp_path, rates, 110)
    assert "FAILED" in rows[2] and "images/s" not in rows[2]


def test_partial_plan_write_is_retryable(tmp_path):
    (tmp_path / "plan.json").write_text('{"status":')
    rows, finished = snapshot(tmp_path, {}, 100)
    assert not finished and "Waiting" in rows[0]
