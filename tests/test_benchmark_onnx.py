import json

import numpy as np
import pytest

from dev.benchmarks.onnx_inference import run

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")


@pytest.fixture
def model_and_inputs(tmp_path):
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("MatMul", ["images", "weight"], ["scores"])],
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


def test_cpu_measurement_records_external_weights_execution_and_outputs(model_and_inputs, tmp_path):
    model, inputs = model_and_inputs
    output = tmp_path / "measurement"
    report = run([model], inputs, output, "CPUExecutionProvider", warmup=1, repeats=2)
    assert report == json.loads((output / "report.json").read_text())
    assert report["status"] == "passed"
    record = report["models"][0]
    assert len(record["files"]) == 2
    assert len(record["seconds"]) == 2
    assert record["median_seconds"] > 0
    assert any(e["op"] == "MatMul" and e["provider"] == "CPUExecutionProvider" for e in record["execution"])
    with np.load(output / "model-0-outputs.npz") as actual, np.load(inputs) as expected:
        np.testing.assert_array_equal(actual["scores"], expected["images"])
    with pytest.raises(FileExistsError):
        run([model], inputs, output, "CPUExecutionProvider")


def test_unavailable_provider_does_not_fall_back(model_and_inputs, tmp_path):
    model, inputs = model_and_inputs
    with pytest.raises(ValueError, match="unavailable"):
        run([model], inputs, tmp_path / "missing", "MissingExecutionProvider")
    assert not (tmp_path / "missing").exists()


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
