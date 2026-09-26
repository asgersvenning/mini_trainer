import os

import numpy as np
import pytest


@pytest.fixture
def tensorrt_add_engine(tmp_path):
    """Small shared engine for opt-in timing and memory tests."""
    if os.environ.get("RUN_CUDA_TESTS") != "1":
        pytest.skip("Set RUN_CUDA_TESTS=1 in an explicitly prepared GPU environment")
    pytest.importorskip("tensorrt")
    import torch

    assert torch.cuda.is_available(), "CUDA requested but unavailable"
    onnx = pytest.importorskip("onnx")
    from dev.benchmarks.inference.tensorrt_build import build

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
    return tmp_path / "build/model.engine", inputs, x, offset
