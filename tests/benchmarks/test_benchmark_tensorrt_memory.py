import json
import sys

import numpy as np
import pytest

from dev.benchmarks.inference.tensorrt_memory import measure


def test_missing_runtime_retains_failure_and_does_not_overwrite(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "tensorrt", None)
    output = tmp_path / "report"
    with pytest.raises(ImportError, match="explicitly prepared"):
        measure("absent.engine", "absent.npz", output)
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "failed" and report["error"].startswith("ImportError:")
    with pytest.raises(FileExistsError):
        measure("absent.engine", "absent.npz", output)


@pytest.mark.parametrize("settings", [{"runs": 0}, {"threads": 0}, {"device": -1}])
def test_invalid_settings_fail_before_output_creation(tmp_path, settings):
    with pytest.raises(ValueError):
        measure("absent.engine", "absent.npz", tmp_path / "report", **settings)
    assert not (tmp_path / "report").exists()


@pytest.mark.parametrize("pinned", [False, True])
def test_real_engine_memory_and_retained_failure(tmp_path, pinned, tensorrt_add_engine):
    engine, inputs, x, offset = tensorrt_add_engine
    result = measure(engine, inputs, tmp_path / "memory", runs=2, pinned=pinned)
    assert result["status"] == "passed"
    assert result == json.loads((tmp_path / "memory/report.json").read_text())
    for stage in ("cuda_initialized", "engine_loaded", "context_and_io", "warm"):
        snapshot = result["memory"][stage]
        assert 0 <= snapshot["device_used_bytes"] <= snapshot["device_total_bytes"]
        assert snapshot["host"]["resident_bytes"] > 0
    with np.load(tmp_path / "memory/outputs.npz") as outputs:
        assert outputs.files == ["scores"]
        np.testing.assert_array_equal(outputs["scores"], x + offset)
    bad_inputs = tmp_path / "wrong.npz"
    np.savez(bad_inputs, wrong=x)
    with pytest.raises(ValueError, match="Input names"):
        measure(engine, bad_inputs, tmp_path / "failed")
    failure = json.loads((tmp_path / "failed/report.json").read_text())
    assert failure["status"] == "failed" and "engine_loaded" in failure["memory"]
