"""Test an ORT CUDA kernel without model assets, PyTorch, or CPU fallback."""

import base64
import os

import numpy as np
import onnxruntime as ort


def main():
    # ONNX opset 18 / IR 10: one FP32 Sigmoid with input/output shape [4].
    # Embedded protobuf avoids requiring the onnx export package for this probe.
    model = base64.b64decode("CAo6RwoPCgF4EgF5IgdTaWdtb2lkEhJjdWRhX3NpZ21vaWRfcHJvYmVaDwoBeBIKCggIARIECgIIBGIPCgF5EgoKCAgBEgQKAggEQgQKABAS")
    print("ORT:", ort.__version__, ort.__file__, flush=True)
    print("JIT settings:", {k: v for k, v in os.environ.items() if k.startswith(("CUDA_FORCE", "CUDA_DISABLE"))}, flush=True)
    ort.preload_dlls()
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    session = ort.InferenceSession(model, sess_options=options, providers=["CUDAExecutionProvider"])
    session.disable_fallback()
    if session.get_providers()[0] != "CUDAExecutionProvider":
        raise RuntimeError("CUDA provider not selected")
    result = session.run(None, {"x": np.array([-1, 0, 1, 2], dtype=np.float32)})[0]
    np.testing.assert_allclose(result, [0.26894142, 0.5, 0.73105858, 0.88079708], rtol=1e-5)
    print("PASS: standalone CUDA Sigmoid", result, flush=True)


if __name__ == "__main__":
    main()
