import json
import subprocess
import sys

import numpy as np
import pytest

from dev.benchmarks.inference.onnx_inference import require_operations, run

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")


@pytest.fixture
def model_and_inputs(tmp_path):
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("MatMul", ["images", "weight"], ["raw_scores"]),
            onnx.helper.make_node("Identity", ["raw_scores"], ["scores"]),
        ],
        "linear",
        [onnx.helper.make_tensor_value_info("images", onnx.TensorProto.FLOAT, ["batch", 2])],
        [onnx.helper.make_tensor_value_info("scores", onnx.TensorProto.FLOAT, ["batch", 2])],
        [onnx.numpy_helper.from_array(np.eye(2, dtype=np.float32), "weight")],
    )
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 18)], ir_version=10)
    path = tmp_path / "model.onnx"
    onnx.save_model(model, path, save_as_external_data=True, all_tensors_to_one_file=True, location="weights.data", size_threshold=0)
    inputs = tmp_path / "inputs.npz"
    np.savez(inputs, images=np.array([[1, 2], [3, 4]], dtype=np.float32))
    return path, inputs


@pytest.mark.parametrize("optimization", ["disable", "basic", "extended", "all"])
def test_cpu_measurement_records_external_weights_execution_and_outputs(model_and_inputs, tmp_path, optimization):
    model, inputs = model_and_inputs
    output = tmp_path / "measurement"
    report = run([model], inputs, output, "CPUExecutionProvider", warmup=1, repeats=2, required_ops=["MatMul"], optimization=optimization)
    assert report == json.loads((output / "report.json").read_text())
    assert report["status"] == "passed"
    assert report["required_ops"] == ["MatMul"]
    assert report["optimization"] == optimization
    record = report["models"][0]
    assert len(record["files"]) == 2
    assert len(record["seconds"]) == 2
    assert record["median_seconds"] > 0
    assert any(e["op"] == "MatMul" and e["provider"] == "CPUExecutionProvider" for e in record["execution"])
    assert any(e["op"] == "Identity" for e in record["execution"]) == (optimization == "disable")
    with np.load(output / "model-0-outputs.npz") as actual, np.load(inputs) as expected:
        np.testing.assert_array_equal(actual["scores"], expected["images"])
    with pytest.raises(FileExistsError):
        run([model], inputs, output, "CPUExecutionProvider")


def test_unavailable_provider_does_not_fall_back(model_and_inputs, tmp_path):
    model, inputs = model_and_inputs
    with pytest.raises(ValueError, match="unavailable"):
        run([model], inputs, tmp_path / "missing", "MissingExecutionProvider")
    assert not (tmp_path / "missing").exists()


def test_missing_required_operation_retains_failed_profile(model_and_inputs, tmp_path):
    model, inputs = model_and_inputs
    output = tmp_path / "missing-operation"
    with pytest.raises(RuntimeError, match="Required operation MatMulInteger"):
        run([model], inputs, output, "CPUExecutionProvider", required_ops=["MatMulInteger"])
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "failed"
    assert report["required_ops"] == ["MatMulInteger"]
    assert report["models"][0]["execution"]
    assert report["models"][0]["seconds"] == []


@pytest.mark.parametrize("gpu_count", [0, 1])
def test_required_operation_rejects_partial_or_complete_cpu_fallback(gpu_count):
    counts = {("Conv", "CUDAExecutionProvider"): 170, ("MatMulInteger", "CPUExecutionProvider"): 2}
    if gpu_count:
        counts[("MatMulInteger", "CUDAExecutionProvider")] = gpu_count
    with pytest.raises(RuntimeError, match="Required operation MatMulInteger"):
        require_operations(counts, "CUDAExecutionProvider", ["Conv", "MatMulInteger"])
    # Auxiliary CPU work is allowed when the required operations remain on GPU.
    require_operations(counts, "CUDAExecutionProvider", ["Conv"])


def test_input_contract_failure_is_retained(model_and_inputs, tmp_path):
    model, inputs = model_and_inputs
    np.savez(inputs, wrong=np.ones((2, 2), dtype=np.float32))
    output = tmp_path / "failure"
    with pytest.raises(ValueError, match="Input names"):
        run([model], inputs, output, "CPUExecutionProvider")
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "failed"
    assert "median_seconds" not in report["models"][0]


def test_advertised_provider_with_only_cpu_execution_fails(model_and_inputs, tmp_path, monkeypatch):
    import onnxruntime as ort

    model, inputs = model_and_inputs
    real_session = ort.InferenceSession
    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])
    # Simulate a provider being advertised but the real graph executing on CPU.
    monkeypatch.setattr(
        ort,
        "InferenceSession",
        lambda path, sess_options, providers: real_session(path, sess_options=sess_options, providers=["CPUExecutionProvider"]),
    )
    output = tmp_path / "fallback"
    with pytest.raises(RuntimeError, match="No profiled operation"):
        run([model], inputs, output, "CUDAExecutionProvider")
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "failed"
    assert all(e["provider"] == "CPUExecutionProvider" for e in report["models"][0]["execution"])
    assert report["models"][0]["seconds"] == []


@pytest.mark.skipif(sys.platform != "linux", reason="Linux resident-memory probe")
@pytest.mark.parametrize("invalid", [False, True])
def test_isolated_cpu_memory_probe_records_measurement_or_failure(model_and_inputs, tmp_path, invalid):
    model, inputs = model_and_inputs
    if invalid:
        np.savez(inputs, wrong=np.ones((2, 2), dtype=np.float32))
    output = tmp_path / "memory"
    command = [
        sys.executable,
        "-m",
        "dev.benchmarks.inference.onnx_cpu_memory",
        "--model",
        str(model),
        "--inputs",
        str(inputs),
        "--output",
        str(output),
        "--warmup",
        "1",
        "--repeats",
        "2",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    report = json.loads((output / "report.json").read_text())
    if invalid:
        assert result.returncode != 0 and report["status"] == "failed"
        assert "Input names" in report["error"]
        assert "median_seconds" not in report and not (output / "outputs.npz").exists()
        return
    assert result.returncode == 0, result.stderr
    assert report["status"] == "measured"
    assert report["session_providers"] == ["CPUExecutionProvider"]
    assert report["session_load_seconds"] > 0 and report["first_run_seconds"] > 0
    assert len(report["seconds"]) == 2 and report["median_seconds"] > 0
    assert len(report["model_files"]) == 2
    snapshots = list(report["memory"].values())
    assert len(snapshots) == 7
    assert all(s["resident_bytes"] > 0 and s["peak_resident_bytes"] > 0 and s["swap_bytes"] >= 0 for s in snapshots)
    peaks = [s["peak_resident_bytes"] for s in snapshots]
    assert peaks == sorted(peaks)
    assert report["environment"]["cpu_affinity"]
    with np.load(output / "outputs.npz") as actual, np.load(inputs) as expected:
        np.testing.assert_array_equal(actual["scores"], expected["images"])
    assert not list(output.glob("*profile*"))
    original_report = (output / "report.json").read_bytes()
    assert subprocess.run(command, capture_output=True, check=False).returncode != 0
    assert (output / "report.json").read_bytes() == original_report


@pytest.mark.skipif(sys.platform != "linux", reason="Linux resident-memory probe")
def test_memory_peak_excludes_parent_pre_exec_allocations():
    child = (
        "import json, resource; from dev.benchmarks.inference.onnx_cpu_memory import resident_memory; "
        "print(json.dumps({'memory': resident_memory(), 'rusage': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}))"
    )
    parent = f"import subprocess, sys; retained = bytearray(128 * 1024**2); subprocess.run([sys.executable, '-c', {child!r}], check=True)"
    result = subprocess.run([sys.executable, "-c", parent], capture_output=True, text=True, check=True)
    report = json.loads(result.stdout)
    assert report["memory"]["peak_resident_bytes"] < report["rusage"] - 64 * 1024**2
