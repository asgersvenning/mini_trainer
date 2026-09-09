"""Architecture-independent export of image models and structured tensor outputs."""

import copy
import hashlib
import json
from collections.abc import Sequence
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory, TemporaryFile
from typing import Any

import numpy as np
import torch
from torch import nn

from mini_trainer.utils import class_path

from .classifier import Classifier
from .context import EmbeddingContext, SupervisionContext


def _flatten(output):
    if isinstance(output, torch.Tensor):
        return [output]
    if isinstance(output, dict):
        return [tensor for value in output.values() for tensor in _flatten(value)]
    if isinstance(output, (list, tuple)):
        return [tensor for value in output for tensor in _flatten(value)]
    if output is None:
        return []
    raise TypeError(f"Export expects tensors or nested lists/tuples/dicts of tensors, found {type(output).__name__}.")


def _structure(output, names):
    if isinstance(output, torch.Tensor):
        return {"tensor": next(names)}
    if isinstance(output, dict):
        if not all(isinstance(key, str) for key in output):
            raise TypeError("Export output dictionaries must have string keys.")
        return {"dict": {key: _structure(value, names) for key, value in output.items()}}
    if isinstance(output, (list, tuple)):
        return {"tuple" if isinstance(output, tuple) else "list": [_structure(value, names) for value in output]}
    if output is None:
        return None
    raise TypeError(f"Unsupported output type: {type(output).__name__}")


def _json_value(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return {"python_type": class_path(value), "repr": repr(value)}


class _TensorOutputs(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images):
        return tuple(_flatten(self.model(images)))


def _dependencies():
    try:
        import onnx
        import onnxruntime
        import onnxscript  # noqa: F401
    except ImportError as error:
        raise ImportError("ONNX export requires optional dependencies. Install mini_trainer[export].") from error
    return onnx, onnxruntime


def _copy_for_export(model, dtype, device="cpu"):
    while isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)) or hasattr(model, "_orig_mod"):
        model = model._orig_mod if hasattr(model, "_orig_mod") else model.module
    model = copy.deepcopy(model).to(device=device, dtype=dtype).eval()
    for module in model.modules():
        if isinstance(module, Classifier):
            module._dirty_cache.clear()
    return model


@contextmanager
def _reference_precision(device):
    """Use full FP32 CUDA arithmetic for the cross-provider reference."""
    if device.type != "cuda":
        yield
        return
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        with torch.autocast("cuda", enabled=False), torch.backends.cudnn.flags(allow_tf32=False):
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32


def export_onnx(
    model: nn.Module,
    example_input: torch.Tensor,
    output_dir: str | Path,
    *,
    preprocessing: dict[str, Any] | None = None,
    dynamic_batch: bool = True,
    verification_inputs: Sequence[torch.Tensor] = (),
    output_names: Sequence[str] | None = None,
    opset_version: int = 18,
    rtol: float = 1e-4,
    atol: float = 1e-5,
    checkpoint_sha256: str | None = None,
    reference_device: str | torch.device = "cpu",
) -> Path:
    """Export the actual eval forward without a backbone or head allowlist.

    example_input is a preprocessed batched floating-point tensor. A private copy
    runs on reference_device in its dtype; the caller's weights, modes, device and caches are
    untouched. DDP/DataParallel/compiled wrappers are unwrapped. Tensor/list/tuple/
    dict outputs are flattened, with structure and classifier metadata in the
    manifest. Dynamic batch is verified at sizes 1, 2 and 4, plus optional real
    verification_inputs. Other dimensions are fixed. Static batch is explicit.

    Preprocessing remains external; supply a JSON-compatible recipe for deployment.
    Unsupported operators propagate exporter errors; models are never substituted.
    Existing output directories are never overwritten.
    Native INT8 training models require a CUDA reference_device and float32 inputs;
    their captured integer forward is verified against ONNX Runtime CPU inference.
    CUDA reference execution disables TF32/autocast temporarily, restoring caller settings.
    """
    if not isinstance(example_input, torch.Tensor) or example_input.ndim < 2 or example_input.shape[0] < 1:
        raise ValueError("example_input must be a nonempty batched tensor.")
    if example_input.dtype not in (torch.float16, torch.float32, torch.float64):
        raise ValueError("Use float16, float32 or float64 example inputs for ONNX Runtime verification.")
    if not np.isfinite([rtol, atol]).all() or min(rtol, atol) < 0:
        raise ValueError("Parity tolerances must be finite and non-negative.")
    if EmbeddingContext.active() or SupervisionContext.get() is not None:
        raise RuntimeError("Export must run outside active training/supervision contexts.")
    destination = Path(output_dir).absolute()
    if destination.exists():
        raise FileExistsError(f"Export destination already exists: {destination}")
    onnx, ort = _dependencies()
    reference_device = torch.device(reference_device)
    native_int8 = any(getattr(parameter, "_is_quantized_training", False) for parameter in model.parameters())
    translations = {}
    if native_int8:
        if reference_device.type != "cuda" or example_input.dtype != torch.float32:
            raise ValueError("Native INT8 ONNX export requires reference_device='cuda' and float32 example inputs.")
        from ._onnx_quantized import scaled_int8_mm

        translations[torch.ops.mini_trainer.scaled_int8_mm.default] = scaled_int8_mm
    sample = example_input.detach().cpu().clone()
    reference = _copy_for_export(model, sample.dtype, reference_device)
    if native_int8:
        # Tensor-subclass decomposition must not assign requires_grad to codes.
        reference.requires_grad_(False)
    with TemporaryFile() as state_file:
        torch.save(reference.state_dict(), state_file)
        state_file.seek(0)
        state_hash = hashlib.file_digest(state_file, "sha256").hexdigest()
    # Batch one is specialized by torch.export; trace at batch two when dynamic.
    trace_input = sample[:1].repeat(2, *([1] * (sample.ndim - 1))) if dynamic_batch else sample
    trace_input = trace_input.to(reference_device)
    with torch.no_grad(), _reference_precision(reference_device):
        observed = reference(trace_input)
    tensors = _flatten(observed)
    if not tensors:
        raise ValueError("The model must return at least one tensor.")
    names = list(output_names) if output_names is not None else [f"output_{i}" for i in range(len(tensors))]
    if (
        len(names) != len(tensors)
        or len(set(names)) != len(names)
        or any(not isinstance(name, str) or not name or name == "images" for name in names)
    ):
        raise ValueError("output_names must contain one unique nonempty name per output tensor.")
    structure = _structure(observed, iter(names))
    classifiers = [
        {"module": name, "type": class_path(module), "metadata": _json_value(module.metadata)}
        for name, module in reference.named_modules()
        if isinstance(module, Classifier)
    ]
    exported = _TensorOutputs(copy.deepcopy(reference)).eval()
    checks = [sample[:1].repeat(batch, *([1] * (sample.ndim - 1))) for batch in (1, 2, 4)] if dynamic_batch else [sample]
    if dynamic_batch:
        checks.append(sample)
    checks.extend(inputs.detach().cpu() for inputs in verification_inputs)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".onnx-export-", dir=destination.parent) as temporary:
        bundle = Path(temporary) / "bundle"
        bundle.mkdir()
        graph = bundle / "model.onnx"
        rng_devices = (
            [reference_device.index if reference_device.index is not None else torch.cuda.current_device()]
            if reference_device.type == "cuda"
            else []
        )
        with torch.no_grad(), torch.random.fork_rng(devices=rng_devices), _reference_precision(reference_device):
            torch.onnx.export(
                exported,
                (trace_input,),
                str(graph),
                input_names=["images"],
                output_names=names,
                opset_version=opset_version,
                dynamo=True,
                dynamic_shapes=({0: torch.export.Dim("batch", min=1)},) if dynamic_batch else None,
                external_data=True,
                custom_translation_table=translations,
            )
        onnx.checker.check_model(str(graph))
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(graph), sess_options=options, providers=["CPUExecutionProvider"])
        errors = []
        for inputs in checks:
            if inputs.dtype != sample.dtype or inputs.shape[1:] != sample.shape[1:]:
                raise ValueError("Verification inputs must match the example's dtype and non-batch dimensions.")
            with torch.inference_mode(), _reference_precision(reference_device):
                expected_output = reference(inputs.to(reference_device))
            if _structure(expected_output, iter(names)) != structure:
                raise ValueError("Model output structure changed during verification.")
            expected = _flatten(expected_output)
            actual = session.run(names, {"images": inputs.numpy()})
            batch_errors = []
            for name, result, value in zip(names, actual, expected, strict=True):
                target = value.detach().cpu().numpy()
                if (
                    result.shape != target.shape
                    or result.dtype != target.dtype
                    or not np.isfinite(result).all()
                    or not np.isfinite(target).all()
                ):
                    raise ValueError(f"ONNX output {name} has an invalid shape, dtype or non-finite values.")
                np.testing.assert_allclose(result, target, rtol=rtol, atol=atol, err_msg=f"ONNX parity failed for {name}")
                batch_errors.append(float(np.max(np.abs(result.astype(np.float64) - target.astype(np.float64)))) if result.size else 0.0)
            errors.append({"batch_size": inputs.shape[0], "max_absolute_error": batch_errors})
        artifacts = {}
        for path in bundle.iterdir():
            if path.is_file():
                with path.open("rb") as artifact:
                    artifacts[path.name] = hashlib.file_digest(artifact, "sha256").hexdigest()
        manifest = {
            "schema_version": 1,
            "architecture": class_path(reference),
            "artifacts": artifacts,
            "source": {"serialized_state_sha256": state_hash, "checkpoint_sha256": checkpoint_sha256},
            "input": {
                "name": "images",
                "dtype": str(sample.dtype).removeprefix("torch."),
                "shape": ["batch" if dynamic_batch else sample.shape[0], *sample.shape[1:]],
            },
            "outputs": [
                {"name": name, "dtype": str(value.dtype).removeprefix("torch."), "example_shape": list(value.shape)}
                for name, value in zip(names, tensors)
            ],
            "output_structure": structure,
            "output_semantics": "model_eval_forward",
            "quantized_training_forward": native_int8,
            "classifiers": classifiers,
            "preprocessing": {"in_graph": False, "recipe": preprocessing, "requires_configuration": preprocessing is None},
            "opset": opset_version,
            "versions": {name: version(name) for name in ("mini_trainer", "torch", "torchvision", "onnx", "onnxscript", "onnxruntime")},
            "verification": {
                "provider": "CPUExecutionProvider",
                "reference_device": str(reference_device),
                "reference_tf32": False if reference_device.type == "cuda" else None,
                "reference_autocast": False if reference_device.type == "cuda" else None,
                "rtol": rtol,
                "atol": atol,
                "cases": errors,
            },
        }
        (bundle / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        if destination.exists():
            raise FileExistsError(f"Export destination already exists: {destination}")
        bundle.rename(destination)
    return destination
