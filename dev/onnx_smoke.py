"""Produce a classifier fixture or verify it without any training dependencies."""

import sys
from pathlib import Path

import numpy as np


def produce(directory):
    import torch

    from mini_trainer.modeling import Classifier
    from mini_trainer.modeling.onnx import export_onnx

    torch.manual_seed(42)
    model, _ = Classifier.build(model_type="resnet18", num_classes=3, model_args={"pretrained": False})
    inputs = torch.randn(3, 3, 32, 32)
    export_onnx(model, inputs, directory / "bundle")
    with torch.no_grad():
        expected = model.eval()(inputs).numpy()
    np.savez(directory / "inputs.npz", images=inputs.numpy(), expected=expected)


def verify(directory):
    import importlib.util
    import json

    import onnxruntime as ort

    assert importlib.util.find_spec("torch") is None
    assert importlib.util.find_spec("mini_trainer") is None
    manifest = json.loads((directory / "bundle/manifest.json").read_text())
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(directory / "bundle/model.onnx"), sess_options=options, providers=["CPUExecutionProvider"])
    with np.load(directory / "inputs.npz") as data:
        actual = session.run([manifest["outputs"][0]["name"]], {"images": data["images"]})[0]
        np.testing.assert_allclose(actual, data["expected"], rtol=1e-4, atol=1e-5)
    print("Standalone ONNX Runtime parity passed without torch or mini_trainer.")


if __name__ == "__main__":
    {"produce": produce, "verify": verify}[sys.argv[1]](Path(sys.argv[2]))
