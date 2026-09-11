import json
import os
import subprocess
import sys

import numpy as np
import pytest

from dev.benchmarks.inference.tensorrt_build import build, compare_outputs, input_profiles


def test_profiles_preserve_named_shapes_and_allow_sample_away_from_optimum():
    feeds = {"images": np.zeros((2, 3, 8, 8)), "offset": np.zeros(3)}
    profiles = input_profiles(feeds)
    profiles["images"] = {"min": [1, 3, 8, 8], "opt": [4, 3, 8, 8], "max": [8, 3, 8, 8]}
    assert input_profiles(feeds, profiles) == profiles
    assert input_profiles({"scalar": np.array(1.0)})["scalar"] == {"min": [], "opt": [], "max": []}


@pytest.mark.parametrize(
    "ranges",
    [
        {"min": [1], "opt": [1], "max": [1]},
        {"min": [3], "opt": [2], "max": [4]},
        {"min": [1], "opt": [5], "max": [4]},
        {"min": [0], "opt": [2], "max": [4]},
        {"min": [True], "opt": [2], "max": [4]},
        {"min": [1.0], "opt": [2], "max": [4]},
        {"min": [1, 1], "opt": [2, 1], "max": [4, 1]},
        {"min": [1], "max": [4]},
    ],
)
def test_profiles_reject_invalid_or_uncovered_shapes(ranges):
    with pytest.raises(ValueError):
        input_profiles({"x": np.zeros(2)}, {"x": ranges})


def test_profiles_require_exact_input_names():
    with pytest.raises(ValueError, match="names"):
        input_profiles({"x": np.zeros(2)}, {"y": {"min": [1], "opt": [2], "max": [4]}})


def test_integer_reference_parity_is_exact_even_above_float_precision():
    actual = {"x": np.array([2**60], dtype=np.int64)}
    expected = {"x": np.array([2**60 + 1], dtype=np.int64)}
    assert compare_outputs(actual, expected, 1, 1)["x"] == {"passed": False, "elements_outside_tolerance": 1}


def test_reference_checks_float_tolerance_and_contract():
    actual = {"x": np.array([1.0, 2.0], dtype=np.float32)}
    expected = {"x": np.array([1.01, 2.0], dtype=np.float32)}
    assert compare_outputs(actual, expected, 0, 0.02)["x"]["passed"]
    assert not compare_outputs(actual, expected, 0, 0.001)["x"]["passed"]
    for wrong in (
        {"y": actual["x"]},
        {"x": np.zeros(2, dtype=np.float64)},
        {"x": np.zeros(3, dtype=np.float32)},
        {"x": np.full(2, np.nan, dtype=np.float32)},
    ):
        with pytest.raises(ValueError):
            compare_outputs(actual, wrong, 0, 0)


def test_help_does_not_import_tensorrt_or_torch():
    code = """
import runpy, sys
sys.argv = ['tensorrt_build', '--help']
try:
    runpy.run_module('dev.benchmarks.inference.tensorrt_build', run_name='__main__')
except SystemExit as error:
    assert error.code == 0
assert 'tensorrt' not in sys.modules
assert 'torch' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


def gpu_dependencies():
    if os.environ.get("RUN_CUDA_TESTS") != "1":
        pytest.skip("Set RUN_CUDA_TESTS=1 in a prepared TensorRT/CUDA environment")
    pytest.importorskip("tensorrt")
    import torch

    if not torch.cuda.is_available():
        pytest.fail("CUDA requested but unavailable")
    return pytest.importorskip("onnx")


@pytest.mark.parametrize("wrong_reference", [False, True])
def test_tensorrt_build_records_execution_and_reference_failure(tmp_path, wrong_reference):
    onnx = gpu_dependencies()
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("MatMul", ["x", "weight"], ["product"]), onnx.helper.make_node("Add", ["product", "offset"], ["scores"])],
        "two-input-linear",
        [
            onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, ["batch", 2]),
            onnx.helper.make_tensor_value_info("offset", onnx.TensorProto.FLOAT, [2]),
        ],
        [onnx.helper.make_tensor_value_info("scores", onnx.TensorProto.FLOAT, ["batch", 2])],
        [onnx.numpy_helper.from_array(np.eye(2, dtype=np.float32), "weight")],
    )
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 18)], ir_version=10)
    path = tmp_path / "model.onnx"
    onnx.save_model(model, path, save_as_external_data=True, all_tensors_to_one_file=True, location="weights.data", size_threshold=0)
    x, offset = np.array([[1, 2], [3, 4]], dtype=np.float32), np.array([0.5, -0.5], dtype=np.float32)
    inputs, reference, output = tmp_path / "inputs.npz", tmp_path / "reference.npz", tmp_path / "result"
    np.savez(inputs, x=x, offset=offset)
    np.savez(reference, scores=x + offset + (1 if wrong_reference else 0))
    profiles = {"x": {"min": [1, 2], "opt": [2, 2], "max": [4, 2]}, "offset": {"min": [2], "opt": [2], "max": [2]}}
    if wrong_reference:
        with pytest.raises(ValueError, match="failed reference parity"):
            build(path, inputs, output, profiles=profiles, reference=reference, optimization=0)
    else:
        build(path, inputs, output, profiles=profiles, reference=reference, optimization=0)
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == ("failed" if wrong_reference else "passed")
    assert report["reference"]["comparison"]["scores"]["passed"] != wrong_reference
    assert len(report["model_files"]) == 2
    assert report["engine"]["bytes"] > 0
    assert json.loads((output / "layers.json").read_text())["Layers"]
    with np.load(output / "outputs.npz") as result:
        np.testing.assert_allclose(result["scores"], x + offset, rtol=0, atol=0)
    with pytest.raises(FileExistsError):
        build(path, inputs, output)


def test_parser_failure_retains_diagnostics_without_engine(tmp_path):
    onnx = gpu_dependencies()
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("MissingBenchmarkPlugin", ["x"], ["y"], domain="mini_trainer.test")],
        "unsupported",
        [onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [2])],
        [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [2])],
    )
    model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 18), onnx.helper.make_opsetid("mini_trainer.test", 1)], ir_version=10
    )
    path, inputs, output = tmp_path / "model.onnx", tmp_path / "inputs.npz", tmp_path / "failure"
    onnx.save(model, path)
    np.savez(inputs, x=np.ones(2, dtype=np.float32))
    with pytest.raises(RuntimeError, match="could not parse"):
        build(path, inputs, output)
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "failed"
    assert report["parser_errors"]
    assert report["messages"]
    assert not (output / "model.engine").exists()
