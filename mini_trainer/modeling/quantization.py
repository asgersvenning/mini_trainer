"""Opt-in PT2E INT8 calibration, QAT and x86 inference.

Preprocessing stays outside the graph. Prepared graphs are training artifacts;
converted reference graphs are deployment artifacts and must be lowered before
claiming integer execution. Nothing in this module changes default training.
"""

import copy
import hashlib
import json
import platform
from collections import Counter
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch import nn

from .classifier import Classifier
from .context import EmbeddingContext, SupervisionContext
from .onnx import _flatten, _json_value, _structure


def _backend():
    try:
        from torchao.quantization.pt2e import export_utils, quantize_pt2e
        from torchao.quantization.pt2e.lowering import lower_pt2e_quantized_to_x86
        from torchao.quantization.pt2e.quantizer.x86_inductor_quantizer import (
            X86InductorQuantizer,
            get_default_x86_inductor_quantization_config,
        )
    except ImportError as error:
        raise ImportError("INT8 quantization requires mini_trainer[quantization].") from error
    return quantize_pt2e, export_utils, X86InductorQuantizer, get_default_x86_inductor_quantization_config, lower_pt2e_quantized_to_x86


def _input(images):
    if not isinstance(images, torch.Tensor) or images.ndim < 2 or images.shape[0] < 1:
        raise ValueError("Supply a nonempty, preprocessed batch tensor.")
    if images.device.type != "cpu" or images.dtype != torch.float32:
        raise ValueError("The initial x86 INT8 profile requires CPU float32 inputs (no autocast).")
    if not torch.isfinite(images).all():
        raise ValueError("Quantization inputs must be finite.")


def _weighted_nodes(graph):
    return [
        n for n in graph.nodes if n.target in (torch.ops.aten.linear.default, torch.ops.aten.conv1d.default, torch.ops.aten.conv2d.default)
    ]


def _graph_mode(graph, training):
    utils = _backend()[1]
    (utils._move_exported_model_to_train if training else utils._move_exported_model_to_eval)(graph)
    # TorchAO switches BatchNorm math, but its exported counter increment stays
    # specialized to training. Switch that increment too, at its graph source.
    for node in graph.graph.nodes:
        if node.target == torch.ops.aten.add_.Tensor:
            target = node.args[0]
            if isinstance(target, torch.fx.Node) and target.op == "get_attr" and str(target.target).endswith("num_batches_tracked"):
                node.args = (target, int(training), *node.args[2:])
    graph.recompile()


class PreparedInt8(nn.Module):
    """A private, captured model with observers or fake quantizers.

    Create optimizers AFTER preparation. Save this object's state_dict along with
    optimizer/scheduler/scaler state; restore into the same preparation recipe.
    QAT train/eval switching covers exported dropout and batchnorm only, not
    arbitrary Python branches or ambient supervision/embedding contexts.
    """

    def __init__(self, graph, recipe):
        super().__init__()
        self.graph = graph
        self.recipe = recipe
        self.register_buffer("observers_frozen", torch.tensor(False))
        self.train(recipe["qat"])

    def train(self, mode=True):
        if not isinstance(mode, bool):
            raise ValueError("training mode must be a bool")
        if mode and not self.recipe["qat"]:
            raise ValueError("PTQ preparation is for calibration; use qat=True for training.")
        self.training = mode
        if self.recipe["qat"]:
            _graph_mode(self.graph, mode)
        return self

    def freeze_observers(self):
        """Keep learned ranges fixed during subsequent QAT steps."""
        self.observers_frozen.fill_(True)

    def forward(self, images):
        _input(images)
        if torch.is_autocast_enabled("cpu"):
            raise ValueError("This QAT/calibration profile uses float32 without AMP.")
        # FakeQuantize.eval() alone does not stop observers. Evaluation must not
        # incorporate held-out examples into training/calibration ranges.
        observers = [m for m in self.graph.modules() if hasattr(m, "observer_enabled")]
        saved = [m.observer_enabled.clone() for m in observers]
        if self.recipe["qat"] and (not self.training or self.observers_frozen):
            for module in observers:
                module.observer_enabled.zero_()
        try:
            return self.graph(images)
        finally:
            for module, enabled in zip(observers, saved):
                module.observer_enabled.copy_(enabled)

    def get_extra_state(self):
        return self.recipe

    def set_extra_state(self, state):
        if state != self.recipe:
            raise ValueError("Quantization checkpoint recipe differs; recreate the same model, shape and QAT configuration.")

    @torch.no_grad()
    def convert(self):
        """Convert an independent copy; keep the training model/optimizer usable."""
        observed = []
        for module in self.graph.modules():
            if hasattr(module, "min_val") and hasattr(module, "max_val"):
                observed.append(module)
                if not module.min_val.numel() or not torch.isfinite(module.min_val).all() or not torch.isfinite(module.max_val).all():
                    raise ValueError("Every quantizer must observe finite training/calibration data before conversion.")
        if not observed:
            raise ValueError("No calibrated quantization observers found.")
        backend, *_ = _backend()
        graph = copy.deepcopy(self.graph)
        if self.recipe["qat"]:
            _graph_mode(graph, False)
        converted = backend.convert_pt2e(graph)
        return Int8Model(converted, copy.deepcopy(self.recipe))


def prepare_int8(model: nn.Module, example_input: torch.Tensor, *, qat=False):
    """Capture the actual model for static W8A8 PTQ or QAT, at a fixed input shape.

    Weights are symmetric per-channel int8; activations are affine per-tensor
    uint8. Bias, normalization and unsupported non-linear operations stay float.
    Graph capture/backend errors propagate. All captured Conv1d/Conv2d/Linear
    operations must receive weight AND activation quantization annotations.
    """
    _input(example_input)
    if not isinstance(qat, bool):
        raise TypeError("qat must be a bool")
    if EmbeddingContext.active() or SupervisionContext.get() is not None:
        raise RuntimeError("Prepare outside embedding/supervision contexts.")
    backend, _, quantizer_cls, config_factory, _ = _backend()
    while isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)) or hasattr(model, "_orig_mod"):
        model = model._orig_mod if hasattr(model, "_orig_mod") else model.module
    with torch.random.fork_rng(devices=[]):
        model = copy.deepcopy(model).cpu().float().eval()
        for module in model.modules():
            if isinstance(module, Classifier):
                module._dirty_cache.clear()
        # Populate masks and immutable evaluation caches outside strict capture.
        with torch.no_grad():
            outputs = model(example_input)
        classifiers = [
            {"module": name, "metadata": _json_value(m.metadata)} for name, m in model.named_modules() if isinstance(m, Classifier)
        ]
        model.train(qat)
        # Strict capture retains functional-op provenance needed by TorchAO's
        # x86 quantizer. Non-strict export silently misses functional linears.
        graph = torch.export.export(model, (example_input,), strict=True).module()
        quantizer = quantizer_cls().set_global(config_factory(is_qat=qat))
        graph = (backend.prepare_qat_pt2e if qat else backend.prepare_pt2e)(graph, quantizer)
    nodes = _weighted_nodes(graph.graph)
    missing = []
    for node in nodes:
        annotation = node.meta.get("quantization_annotation")
        if annotation is None or sum(spec is not None for spec in annotation.input_qspec_map.values()) < 2:
            missing.append(node.name)
    if not nodes or missing:
        raise ValueError(
            f"INT8 requires quantized weights and activations for captured Conv/Linear operations; missing: {missing or 'all'}"
        )
    recipe = {
        "format_version": 1,
        "backend": "x86_inductor",
        "qat": qat,
        "torch": str(torch.__version__),
        "torchao": version("torchao"),
        "input_shape": list(example_input.shape),
        "weight_bits": 8,
        "activation_bits": 8,
        "weight_dtype": "int8",
        "activation_dtype": "uint8",
        "classifiers": classifiers,
        "output_structure": _structure(outputs, iter(f"output_{i}" for i in range(len(_flatten(outputs))))),
        "weighted_operations": [{"name": n.name, "operator": str(n.target)} for n in nodes],
    }
    return PreparedInt8(graph, recipe)


class Int8Model:
    """Converted reference graph with explicit x86 lowering and portable storage."""

    def __init__(self, graph, recipe):
        self.graph = graph
        self.recipe = recipe

    @torch.no_grad()
    def lower(self, example_input):
        """Return an inference graph with verified oneDNN INT8 Conv/Linear kernels.

        Reference Q/DQ execution is not integer arithmetic. Require the lowered
        graph to contain integer kernels and no residual floating Conv/Linear.
        """
        _input(example_input)
        if platform.machine().lower() not in ("x86_64", "amd64"):
            raise RuntimeError("This quantization backend requires x86 CPU hardware.")
        lowered = _backend()[4](copy.deepcopy(self.graph), (example_input,))
        operators = Counter(str(n.target) for n in lowered.graph.nodes if n.op == "call_function")
        integer = {op: count for op, count in operators.items() if op.startswith("onednn.q") and "pointwise" in op}
        floating = [
            op
            for op in operators
            if op.startswith(("aten.linear.", "aten.convolution.", "aten.conv1d.", "aten.conv2d.", "aten.mm.", "aten.addmm."))
        ]
        if not integer or floating:
            raise RuntimeError(f"Incomplete INT8 lowering: integer kernels={integer}, remaining floating kernels={floating}")
        actual = lowered(example_input)
        expected = self.graph(example_input)
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
        return lowered, {"integer_kernels": integer, "operators": dict(operators)}

    @torch.no_grad()
    def save(self, output_dir, example_input, *, preprocessing, calibration):
        """Save a reference .pt2 graph and manifest after verifying native lowering.

        Supply JSON provenance for preprocessing and training-only calibration.
        Destination must be new. Packed oneDNN weights are rebuilt on load.
        """
        _input(example_input)
        destination = Path(output_dir).absolute()
        if destination.exists():
            raise FileExistsError(destination)
        _, coverage = self.lower(example_input)
        manifest = {"recipe": self.recipe, "preprocessing": preprocessing, "calibration": calibration, "lowering": coverage}
        json.dumps(manifest, allow_nan=False)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=".int8-", dir=destination.parent) as temporary:
            bundle = Path(temporary) / "bundle"
            bundle.mkdir()
            path = bundle / "model.pt2"
            program = torch.export.export(self.graph, (example_input,))
            torch.export.save(program, path)
            restored = torch.export.load(path).module()
            torch.testing.assert_close(restored(example_input), self.graph(example_input), rtol=0, atol=0)
            manifest["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            manifest["artifact_bytes"] = path.stat().st_size
            (bundle / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
            bundle.rename(destination)
        return destination


def load_int8(output_dir):
    """Load a saved reference graph; call .lower(example_input) for integer execution."""
    _backend()  # Register decomposed quantization operators before torch.export.load.
    directory = Path(output_dir)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["recipe"]["format_version"] != 1:
        raise ValueError("Unsupported INT8 bundle version")
    path = directory / "model.pt2"
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["sha256"]:
        raise ValueError("INT8 artifact checksum mismatch")
    return Int8Model(torch.export.load(path).module(), manifest["recipe"])
