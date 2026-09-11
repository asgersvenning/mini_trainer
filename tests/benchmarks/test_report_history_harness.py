import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture
def run_history(tmp_path):
    results = tmp_path / "results with spaces"
    results.mkdir()
    (results / "status.json").write_text('{"phase":"preflight","exit_code":19}')
    history = tmp_path / "history"
    env = {
        **os.environ,
        "BENCHMARK_PYTHON": sys.executable,
        "BENCHMARK_RUN_ID": "trt-123-2",
        "BENCHMARK_REVISION": "a" * 40,
        "BENCHMARK_PROFILE": "Blair $(not-a-command) / GPU",
        "BENCHMARK_NOTE": "Uncontended target",
        "BENCHMARK_RUN_URL": "https://github.com/owner/repo/actions/runs/123",
        "BENCHMARK_PERFORMANCE_VALID": "false",
    }

    def run():
        return subprocess.run(
            ["bash", "dev/check-report-history.sh", str(results), str(history)],
            cwd=Path(__file__).resolve().parents[2],
            env=env,
            capture_output=True,
            text=True,
        )

    return run, results, history, env


def test_failed_target_produces_idempotent_compact_artifact(run_history):
    run, _, history, env = run_history
    env["BENCHMARK_PERFORMANCE_VALID"] = "true"
    result = run()
    assert result.returncode == 0, result.stderr
    record = history / "records/trt-123-2.json"
    original = record.read_bytes()
    data = json.loads(original)
    assert data["profile"] == env["BENCHMARK_PROFILE"]
    assert not data["performance_valid"]
    assert "Quality results unavailable" in (history / "index.html").read_text()
    assert run().returncode == 0 and record.read_bytes() == original


def test_malformed_evaluation_is_not_hidden_by_failure_status(run_history):
    run, results, history, _ = run_history
    (results / "evaluation").mkdir()
    (results / "evaluation/report.json").write_text("invalid json")
    assert run().returncode != 0
    assert not history.exists()


@pytest.mark.parametrize("value", ["yes", "1", "TRUE"])
def test_invalid_performance_declaration_fails(run_history, value):
    run, _, history, env = run_history
    env["BENCHMARK_PERFORMANCE_VALID"] = value
    result = run()
    assert result.returncode == 2 and "must be true or false" in result.stderr
    assert not history.exists()
