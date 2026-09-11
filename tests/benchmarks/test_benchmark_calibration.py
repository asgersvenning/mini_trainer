import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from dev.benchmarks.inference.onnx_calibration import calibrate, calibration_manifest, load_batch


@pytest.fixture
def inputs(tmp_path):
    entries = []
    for index, x in enumerate(([[0, 1], [1, 2]], [[-4, 10], [6, 20]])):
        path = tmp_path / f"batch-{index}.npz"
        np.savez(path, x=np.array(x, dtype=np.float32), offset=np.array([0.5, -0.5], dtype=np.float32))
        entries.append(
            {
                "path": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "sample_ids": [f"train-{2 * index}", f"train-{2 * index + 1}"],
            }
        )
    metadata = {
        "schema_version": 1,
        "split": "train",
        "batch_input": "x",
        "provenance": {"dataset": "deterministic fixture", "preprocessing": "float features"},
        "batches": entries,
    }
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(metadata))
    return path, metadata


@pytest.mark.parametrize("split", ["val", "test", ""])
def test_calibration_rejects_declared_heldout_or_unknown_split(inputs, split):
    path, metadata = inputs
    metadata["split"] = split
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="train/calibration"):
        calibration_manifest(path)


def test_manifest_rejects_duplicate_sample_ids(inputs):
    path, metadata = inputs
    metadata["batches"][1]["sample_ids"][0] = metadata["batches"][0]["sample_ids"][0]
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="unique"):
        calibration_manifest(path)


def test_batch_identity_hash_and_dimensions_are_checked(inputs):
    path, metadata = inputs
    batch = metadata["batches"][0]
    feeds, record = load_batch(path, metadata, batch)
    assert record["sha256"] == batch["sha256"]
    assert feeds["x"].shape == (2, 2) and feeds["offset"].shape == (2,)
    with pytest.raises(ValueError, match="hash mismatch"):
        load_batch(path, metadata, {**batch, "sha256": "wrong"})
    with pytest.raises(ValueError, match="leading dimension"):
        load_batch(path, metadata, {**batch, "sample_ids": ["only-one"]})


@pytest.fixture
def model(tmp_path):
    onnx = pytest.importorskip("onnx")
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("Gemm", ["x", "weight", "bias"], ["product"]),
            onnx.helper.make_node("Add", ["product", "offset"], ["scores"]),
        ],
        "linear",
        [
            onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, ["batch", 2]),
            onnx.helper.make_tensor_value_info("offset", onnx.TensorProto.FLOAT, [2]),
        ],
        [onnx.helper.make_tensor_value_info("scores", onnx.TensorProto.FLOAT, ["batch", 2])],
        [
            onnx.numpy_helper.from_array(np.eye(2, dtype=np.float32), "weight"),
            onnx.numpy_helper.from_array(np.array([0.25, -0.25], dtype=np.float32), "bias"),
        ],
    )
    parent = tmp_path / "source"
    parent.mkdir()
    path = parent / "model.onnx"
    onnx.save_model(
        onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 18)], ir_version=10),
        path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="weights.data",
        size_threshold=0,
    )
    return path


@pytest.mark.parametrize("method", ["minmax", "percentile"])
@pytest.mark.parametrize("signed", [False, True])
def test_calibration_recipes_keep_all_batches_source_and_thread_limits(model, inputs, tmp_path, monkeypatch, method, signed):
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    from onnxruntime.quantization.calibrate import load_tensors_data

    manifest, metadata = inputs
    output = tmp_path / "output"
    original = {p.name: p.read_bytes() for p in model.parent.iterdir()}
    sessions, inference_paths = [], []
    real_session, real_infer = ort.InferenceSession, onnx.shape_inference.infer_shapes_path

    def session(*args, **kwargs):
        options = kwargs["sess_options"]
        sessions.append((options.intra_op_num_threads, options.inter_op_num_threads))
        return real_session(*args, **kwargs)

    def infer(source, target, *args, **kwargs):
        inference_paths.append(Path(target))
        return real_infer(source, target, *args, **kwargs)

    monkeypatch.setattr(ort, "InferenceSession", session)
    monkeypatch.setattr(onnx.shape_inference, "infer_shapes_path", infer)
    report = calibrate(
        model,
        manifest,
        output,
        method=method,
        activation_type="int8" if signed else "uint8",
        symmetric_activations=signed,
        float_bias=signed,
        threads=2,
    )
    assert report["status"] == "passed"
    assert report == json.loads((output / "report.json").read_text())
    assert sessions == [(2, 1), (2, 1)]
    assert inference_paths and all(output in p.parents for p in inference_paths)
    assert {p.name: p.read_bytes() for p in model.parent.iterdir()} == original
    assert [b["sample_ids"] for b in report["batches"]] == [b["sample_ids"] for b in metadata["batches"]]
    assert report["output_ops"]["QuantizeLinear"] > 0
    if method == "minmax":
        low, high = load_tensors_data(output / "ranges.json")["x"].range_value
        np.testing.assert_array_equal(low, np.array(-4, dtype=np.float32))
        np.testing.assert_array_equal(high, np.array(20, dtype=np.float32))
    graph = onnx.load(output / "model.onnx")
    initializers = {x.name: x for x in graph.graph.initializer}
    assert initializers["x_zero_point"].data_type == (onnx.TensorProto.INT8 if signed else onnx.TensorProto.UINT8)
    if signed:
        assert not onnx.numpy_helper.to_array(initializers["x_zero_point"]).any()
        assert initializers["bias"].data_type == onnx.TensorProto.FLOAT
    else:
        assert initializers["bias_quantized"].data_type == onnx.TensorProto.INT32
    with np.load(output / "calibration-smoke.npz") as smoke:
        assert np.isfinite(smoke["scores"]).all()
        assert smoke["scores"].shape == (2, 2)
    with pytest.raises(FileExistsError):
        calibrate(model, manifest, output)


def test_calibration_batch_failure_retains_failed_report(model, inputs, tmp_path):
    pytest.importorskip("onnxruntime")
    path, metadata = inputs
    metadata["batches"][1]["sha256"] = "changed"
    path.write_text(json.dumps(metadata))
    output = tmp_path / "failed"
    with pytest.raises(ValueError, match="hash mismatch"):
        calibrate(model, path, output, method="minmax")
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "failed"
    assert len(report["batches"]) == 1
    assert not (output / "model.onnx").exists()
