"""Export a completed floating-point run and retain real validation inputs for ORT."""

import argparse
import json
from pathlib import Path

from compare import digest, write_json


def main():
    import numpy as np
    import onnxruntime as ort
    import torch

    from mini_trainer.data.io import make_read_and_resize_fn
    from mini_trainer.modeling import Classifier
    from mini_trainer.modeling.onnx import export_onnx

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comparison", type=Path)
    parser.add_argument("run")
    args = parser.parse_args()
    root = args.comparison.resolve()
    run = root / "runs" / args.run
    result = json.loads((run / "result.json").read_text())
    if result["status"] != "completed" or digest(result["checkpoint"]) != result["checkpoint_sha256"]:
        raise ValueError("Require a completed run with an unchanged checkpoint")
    if result["options"].get("quantized_training"):
        raise ValueError("This FP32 follow-up excludes native INT8; use the explicit native INT8 export workflow in docs/onnx.md")
    prepared = json.loads((root / "prepared.json").read_text())
    if digest(root / "data_index.json") != prepared["data_index.json"]:
        raise ValueError("Frozen dataset index changed")
    config = json.loads((root / "comparison.json").read_text())
    data = json.loads((root / "data_index.json").read_text())
    paths = [p for p, split in zip(data["path"], data["split"], strict=True) if split == "validation"][:4]
    if len(paths) < 4:
        raise ValueError("Need four real validation images for dynamic-batch verification")
    model, preprocess = Classifier.build(
        weights=result["checkpoint"],
        device="cpu",
        dtype=torch.float32,
        model_args={"pretrained": False},
    )
    model.eval()
    reader = make_read_and_resize_fn((config["size"], config["size"]), torch.device("cpu"), torch.uint8)
    with torch.inference_mode():
        images = preprocess(torch.stack([reader(p) for p in paths]))
        eager = model(images)
    destination = run / "onnx"
    export_onnx(
        model,
        images[:2],
        destination,
        verification_inputs=[images[:1], images],
        preprocessing={
            "loader": "mini_trainer.data.io.make_read_and_resize_fn",
            "size": [config["size"], config["size"]],
            "loader_dtype": "uint8",
            "model_preprocess": repr(preprocess),
            "example_image_sha256": {p: digest(p) for p in paths},
            "description": "RGB decode and repository resize, then checkpoint preprocessor; preprocessing remains outside ONNX",
        },
        checkpoint_sha256=result["checkpoint_sha256"],
    )
    manifest = json.loads((destination / "manifest.json").read_text())
    session = ort.InferenceSession(str(destination / "model.onnx"), providers=["CPUExecutionProvider"])
    outputs = session.run(None, {manifest["input"]["name"]: images.numpy()})
    if len(outputs) != len(eager):
        raise ValueError("Output count differs")
    for expected, actual in zip(eager, outputs, strict=True):
        np.testing.assert_allclose(actual, expected.numpy(), rtol=1e-4, atol=1e-5)
    np.savez(destination / "validation-example.npz", images=images.numpy(), **{f"output_{i}": v for i, v in enumerate(outputs)})
    write_json(
        destination / "real-image-parity.json",
        {
            "status": "passed",
            "images": paths,
            "checkpoint_sha256": result["checkpoint_sha256"],
            "provider": "CPUExecutionProvider",
            "rtol": 1e-4,
            "atol": 1e-5,
            "scope": "Four validation images; dynamic-batch FP32 export parity, not held-out quality or target-GPU speed",
        },
    )


if __name__ == "__main__":
    main()
