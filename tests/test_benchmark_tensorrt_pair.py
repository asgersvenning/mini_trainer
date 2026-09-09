import json
import os
import subprocess
import sys

import numpy as np
import pytest

from dev.benchmarks.tensorrt_pair import benchmark, paired_trials, summarize


@pytest.mark.parametrize("reverse", [False, True])
def test_measurements_alternate_and_exclude_warmup(reverse):
    calls = []

    def execute(name):
        calls.append(name)
        return float(len(calls))

    trials = list(paired_trials(execute, 1, 3, reverse))
    assert len(calls) == 8 and len(trials) == 3
    expected = ["baseline", "candidate", "candidate", "baseline"] * 2
    assert calls == (["candidate" if x == "baseline" else "baseline" for x in expected] if reverse else expected)
    assert [t["index"] for t in trials] == [0, 1, 2]
    assert min(trials[0]["seconds"].values()) == 3


def test_summary_uses_paired_ratios_not_ratio_of_medians():
    trials = [{"seconds": {"baseline": a, "candidate": b}} for a, b in [(1, 2), (2, 200), (100, 100)]]
    result = summarize(trials)
    assert result["median_paired_ratio"] == 2
    assert result["candidate_over_baseline_ratios"] == [2, 100, 1]
    assert result["median_seconds"] == {"baseline": 2, "candidate": 100}


@pytest.mark.parametrize("duration", [0, -1, float("nan"), float("inf")])
def test_invalid_clock_readings_are_rejected(duration):
    with pytest.raises(ValueError, match="finite and positive"):
        list(paired_trials(lambda _: duration, 0, 1))


def test_complete_trials_survive_later_execution_failure():
    trials = []
    times = iter([1, 2, 3])

    def execute(_):
        return next(times, 0)

    with pytest.raises(ValueError):
        trials.extend(paired_trials(execute, 0, 2))
    assert len(trials) == 1


def test_help_does_not_load_gpu_libraries():
    code = """
import runpy, sys
sys.argv = ['tensorrt_pair', '--help']
try:
    runpy.run_module('dev.benchmarks.tensorrt_pair', run_name='__main__')
except SystemExit as error:
    assert error.code == 0
assert 'tensorrt' not in sys.modules and 'torch' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True)


@pytest.mark.parametrize("pinned", [False, True])
def test_real_engine_pair_preserves_named_outputs_and_failures(tmp_path, pinned):
    if os.environ.get("RUN_CUDA_TESTS") != "1":
        pytest.skip("Set RUN_CUDA_TESTS=1 in an explicitly prepared GPU environment")
    pytest.importorskip("tensorrt")
    import torch

    assert torch.cuda.is_available(), "CUDA requested but unavailable"
    onnx = pytest.importorskip("onnx")
    from dev.benchmarks.tensorrt_build import build

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Add", ["x", "offset"], ["scores"])],
        "two-input",
        [
            onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [2, 2]),
            onnx.helper.make_tensor_value_info("offset", onnx.TensorProto.FLOAT, [2]),
        ],
        [onnx.helper.make_tensor_value_info("scores", onnx.TensorProto.FLOAT, [2, 2])],
    )
    model, inputs = tmp_path / "model.onnx", tmp_path / "inputs.npz"
    onnx.save(onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 18)], ir_version=10), model)
    x, offset = np.arange(4, dtype=np.float32).reshape(2, 2), np.array([0.5, -0.5], dtype=np.float32)
    np.savez(inputs, x=x, offset=offset)
    build(model, inputs, tmp_path / "build", optimization=0)
    engine = tmp_path / "build/model.engine"
    output = tmp_path / "pair"
    report = benchmark(engine, engine, inputs, output, warmup=1, repeats=3, pinned=pinned)
    assert report["status"] == "passed" and len(report["trials"]) == 3
    assert report == json.loads((output / "report.json").read_text())
    assert report["models"]["baseline"]["sha256"] == report["models"]["candidate"]["sha256"]
    for mode in ("baseline", "candidate"):
        with np.load(output / f"{mode}-outputs.npz") as result:
            np.testing.assert_array_equal(result["scores"], x + offset)
    with pytest.raises(FileExistsError):
        benchmark(engine, engine, inputs, output)
    bad = tmp_path / "bad.engine"
    bad.write_bytes(b"not an engine")
    failed = tmp_path / "failure"
    with pytest.raises(RuntimeError, match="deserialize candidate"):
        benchmark(engine, bad, inputs, failed, warmup=0, repeats=1)
    retained = json.loads((failed / "report.json").read_text())
    assert retained["status"] == "failed" and retained["messages"]
