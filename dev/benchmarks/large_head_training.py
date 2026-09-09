"""Synthetic full-model or frozen-backbone training capacity probe; not convergence."""

import hashlib
import json
import platform
import statistics
import time
from argparse import ArgumentParser
from pathlib import Path

import torch

from mini_trainer.builders import BaseBuilder
from mini_trainer.hierarchical.model import HierarchicalClassifier
from mini_trainer.modeling import Classifier
from mini_trainer.modeling.classifier import classification_module
from mini_trainer.modeling.quantized_training import prepare_quantized_training
from mini_trainer.trainer import _optimizer_step
from mini_trainer.training import MuonAuxAdamW
from mini_trainer.training.compilation import compile_optimizer as prepare_compiled_optimizer
from mini_trainer.training.compilation import validate_optimizer_compilation


def run(
    output,
    classes=10000,
    hierarchical=False,
    frozen=False,
    quantized=False,
    batch_size=32,
    image_size=128,
    seed=42,
    warmup=3,
    steps=5,
    device="cuda:0",
    dtype="float16",
    backbone="efficientnet_v2_s",
    compile_optimizer=False,
    optimizer_cudagraphs=False,
):
    device = torch.device(device)
    validate_optimizer_compilation(compile_optimizer, optimizer_cudagraphs, device)
    if min(classes, batch_size, image_size, warmup, steps) < 1 or classes < 2 or batch_size < 2:
        raise ValueError("Require at least two classes/samples and positive image size, warmup and steps")
    if device.type not in ("cpu", "cuda") or dtype not in ("float32", "float16", "bfloat16"):
        raise ValueError("Unsupported device or dtype")
    if device.type == "cpu" and (quantized or dtype != "float32"):
        raise ValueError("CPU is a float32 diagnostic only; native INT8 training requires CUDA")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("An accessible CUDA GPU is required")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    report = {
        "schema_version": 1,
        "status": "running",
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "settings": {
            "classes": classes,
            "hierarchical": hierarchical,
            "frozen_backbone": frozen,
            "quantized": quantized,
            "batch_size": batch_size,
            "image_size": image_size,
            "seed": seed,
            "warmup": warmup,
            "steps": steps,
            "device": str(device),
            "dtype": dtype,
            "backbone": backbone,
            "hidden": "symmetric",
            "normalized": True,
            "compile_optimizer": compile_optimizer,
            "optimizer_cudagraphs": optimizer_cudagraphs,
        },
        "environment": {"platform": platform.platform(), "torch": torch.__version__},
        "warmup": [],
        "steps": [],
        "scope": (
            "Repeated fixed synthetic uint8 images and labels, MuonAuxAdamW and AMP. "
            "Optimizer compilation/graph flags describe requested configuration, not verified graph replay. "
            "Full-model or frozen-backbone updates; no cached embeddings. "
            "No convergence, loader, checkpoint, distributed or target-hardware performance claims."
        ),
    }

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    try:
        if device.type == "cuda":
            torch.cuda.set_device(device)
            torch.cuda.reset_peak_memory_stats(device)
            report["environment"]["gpu"] = torch.cuda.get_device_name(device)
        # Independent CPU generator prevents quantization/setup RNG consumption
        # from changing the synthetic workload between float and INT8 runs.
        generator = torch.Generator().manual_seed(seed + 1)
        images = torch.randint(0, 256, (batch_size, 3, image_size, image_size), generator=generator, dtype=torch.uint8)
        labels = torch.randint(classes, (batch_size,), generator=generator)
        report["input_sha256"] = hashlib.sha256(images.numpy().tobytes() + labels.numpy().tobytes()).hexdigest()
        images, labels = images.to(device), labels.to(device)
        cls = HierarchicalClassifier if hierarchical else Classifier
        kwargs = {"sparse_masks": [torch.arange(classes, device=device) // 100]} if hierarchical else {}
        model, preprocess = cls.build(
            model_type=backbone,
            num_classes=classes,
            hidden=True,
            normalized=True,
            model_args={"pretrained": False},
            device=device,
            resize_size=image_size,
            **kwargs,
        )
        head = classification_module(model)
        report["head_trainable_parameters"] = sum(p.numel() for p in head.parameters() if p.requires_grad)
        head_ids = {id(p) for p in head.parameters()}
        if frozen:
            for parameter in model.parameters():
                if id(parameter) not in head_ids:
                    parameter.requires_grad_(False)
            # Do not retain the final floating parameter after INT8 preparation
            # replaces it; a large classification head can dominate this probe.
            del parameter
            model.eval()
            head.train()
        else:
            model.train()
        report["recipe"] = prepare_quantized_training(model) if quantized else None
        if sum(p.numel() for p in head.parameters() if p.requires_grad) != report["head_trainable_parameters"]:
            raise RuntimeError("Freezing/preparation changed the head's trainable parameter contract")
        optimizer = BaseBuilder.build_optimizer(model, MuonAuxAdamW, lr=0.01, weight_decay=0.0)
        if compile_optimizer:
            prepare_compiled_optimizer(optimizer, cudagraphs=optimizer_cudagraphs)
        scaler = BaseBuilder.build_scaler(device.type, enabled=device.type == "cuda" and dtype == "float16")
        report["parameters"] = {
            "total": sum(p.numel() for p in model.parameters()),
            "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        }
        report["parameter_bytes"] = sum(
            p.int_data.numel() + p.scale.numel() * p.scale.element_size()
            if getattr(p, "_is_quantized_training", False)
            else p.numel() * p.element_size()
            for p in model.parameters()
        )
        torch.manual_seed(seed + 2)

        def step():
            with torch.autocast(device.type, dtype=getattr(torch, dtype), enabled=dtype != "float32"):
                scores = model(preprocess(images))
                values = scores if isinstance(scores, (tuple, list)) else [scores]
                loss = torch.nn.functional.cross_entropy(values[0], labels)
                if hierarchical:
                    loss = loss + torch.nn.functional.cross_entropy(values[1], labels // 100)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            updated = _optimizer_step(optimizer, scaler)
            return loss.detach(), updated

        for _ in range(warmup):
            loss, updated = step()
            report["warmup"].append({"loss": float(loss), "updated": updated})
        synchronize()
        if device.type == "cuda":
            report["setup_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
        for _ in range(steps):
            synchronize()
            started = time.perf_counter()
            loss, updated = step()
            synchronize()
            elapsed = time.perf_counter() - started
            report["steps"].append({"seconds": elapsed, "loss": float(loss), "updated": updated})
        if device.type == "cuda":
            report["measured_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
            report["measured_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
        if any(not s["updated"] or not torch.isfinite(torch.tensor(s["loss"])) for s in report["steps"]):
            raise RuntimeError("Measured updates were skipped or nonfinite; timings are not a successful training comparison")
        # Some valid heads retain trainable parameters for inactive branches
        # (for example BatchNorm when normalization selects unit embeddings).
        # Record those explicitly, while requiring an active head gradient.
        report["unused_trainable_parameters"] = [name for name, p in model.named_parameters() if p.requires_grad and p.grad is None]
        report["parameters"]["with_gradient"] = sum(p.numel() for p in model.parameters() if p.grad is not None)
        if not any(p.requires_grad and p.grad is not None for p in head.parameters()):
            raise RuntimeError("The classification head did not receive gradients")
        if frozen and any(p.grad is not None for p in model.parameters() if not p.requires_grad):
            raise RuntimeError("A frozen parameter received a gradient")
        report["median_seconds_per_update"] = statistics.median(s["seconds"] for s in report["steps"])
        report["optimizer_steps"] = optimizer._step_count
        report["status"] = "measured"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    for name, default in (("classes", 10000), ("batch-size", 32), ("image-size", 128), ("seed", 42), ("warmup", 3), ("steps", 5)):
        parser.add_argument("--" + name, type=int, default=default)
    for flag in ("hierarchical", "frozen", "quantized", "compile-optimizer", "optimizer-cudagraphs"):
        parser.add_argument("--" + flag, action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float16")
    parser.add_argument("--backbone", default="efficientnet_v2_s")
    run(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
