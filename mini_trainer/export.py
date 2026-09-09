"""Export trained models using the existing architecture and classifier loaders."""

import hashlib
import json
from argparse import ArgumentParser
from pathlib import Path

import torch

from mini_trainer.modeling import Classifier, classification_module
from mini_trainer.modeling.architectures.load import get_dynamic_model, resolve_backbone_getter
from mini_trainer.modeling.onnx import export_onnx
from mini_trainer.modeling.quantized_training import load_training_weights


def main(
    weights: str,
    output: str,
    input_shape: list[int] | None = None,
    preprocessing: dict | None = None,
    model_args: dict | None = None,
    dynamic_batch: bool = True,
    batch_size: int = 2,
    reference_device: str = "cpu",
):
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    with Path(weights).open("rb") as handle:
        checkpoint_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    state = load_training_weights(weights, map_location="cpu")
    state = state.get("model", state)
    metadata = Classifier.extract_metadata(state)
    model_type = metadata.get("backbone_class")
    if not model_type:
        raise ValueError("Checkpoint is missing architecture metadata; use export_onnx with an instantiated model.")
    getter, _ = resolve_backbone_getter(model_type)
    build_args = {} if getter is get_dynamic_model else {"pretrained": False}
    build_args.update(model_args or {})
    model, preprocess = Classifier.build(weights=state, device="cpu", dtype=torch.float32, model_args=build_args)
    if input_shape is None:
        size = classification_module(model).metadata.get("resize_size", 224)
        width, height = (size, size) if isinstance(size, int) else size
        with torch.no_grad():
            sample = preprocess(torch.zeros(batch_size, 3, height, width, dtype=torch.uint8)).float()
    else:
        if not input_shape or any(size < 1 for size in input_shape):
            raise ValueError("input_shape must contain positive non-batch dimensions.")
        sample = torch.zeros(batch_size, *input_shape)
    return export_onnx(
        model,
        sample,
        output,
        preprocessing=preprocessing,
        dynamic_batch=dynamic_batch,
        checkpoint_sha256=checkpoint_hash,
        reference_device=reference_device,
    )


def run():
    parser = ArgumentParser(description="Export a trained image model as a verified ONNX bundle.")
    parser.add_argument("--weights", required=True, help="mini_trainer .pt weights or .pth training checkpoint.")
    parser.add_argument("--output", required=True, help="New directory for the ONNX bundle and manifest.")
    parser.add_argument("--input-shape", type=int, nargs="+", help="Preprocessed input dimensions, excluding batch (usually C H W).")
    parser.add_argument("--preprocessing", type=Path, help="JSON deployment recipe for preprocessing outside the graph.")
    parser.add_argument("--model-args", type=json.loads, help="JSON constructor arguments for the existing architecture loader.")
    parser.add_argument("--static-batch", action="store_false", dest="dynamic_batch", help="Export a fixed batch size.")
    parser.add_argument("--batch-size", type=int, default=2, help="Example/static batch size (default: 2).")
    parser.add_argument("--reference-device", default="cpu", help="PyTorch parity device; native INT8 training requires cuda.")
    args = vars(parser.parse_args())
    if args["preprocessing"] is not None:
        args["preprocessing"] = json.loads(args["preprocessing"].read_text())
    print(main(**args))


if __name__ == "__main__":
    run()
