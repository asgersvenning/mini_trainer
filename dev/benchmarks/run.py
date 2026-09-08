"""Run reproducible dataset benchmarks through the repository's training pipeline."""

import hashlib
import json
import os
import platform
import subprocess
import time
from argparse import ArgumentParser
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import matplotlib
import numpy as np
import torch

from mini_trainer.data import get_inference_dataloader
from mini_trainer.hierarchical.integration import HierarchicalBuilder
from mini_trainer.hierarchical.model import HierarchicalClassifier
from mini_trainer.modeling import Classifier
from mini_trainer.train import main as train
from mini_trainer.training import MuonAuxAdamW

from .datasets import prepare_real
from .models import NoAugmentationBuilder
from .synthetic import generate


class HierarchicalBenchmarkBuilder(HierarchicalBuilder):
    build_augmentation = staticmethod(NoAugmentationBuilder.build_augmentation)


def run(
    output: str | Path,
    seed: int = 42,
    epochs: int = 12,
    *,
    device: str = "cpu",
    dtype: str = "float32",
    cache: str = "NONE",
    num_workers: int = 0,
    dataset: str = "synthetic",
    data_root: str | Path | None = None,
    class_spec: str | Path | None = None,
    quantized_training: bool = False,
    compile: bool = False,
    hidden: int = 0,
    batch_size: int = 32,
    cache_workers: int | None = None,
):
    matplotlib.use("Agg", force=True)
    cache = "CPU" if cache == "RAM" else cache
    target_device = torch.device(device)
    precision = getattr(torch, dtype)
    if target_device.type not in ("cpu", "cuda") or dtype not in ("float32", "float16", "bfloat16"):
        raise ValueError("Benchmark profiles support CPU/CUDA with float32, float16 or bfloat16.")
    if target_device.type == "cpu" and (dtype == "float16" or cache == "CUDA"):
        raise ValueError("Use CUDA for the float16 or CUDA-cache profiles.")
    if num_workers < 0 or epochs < 1 or hidden < 0 or batch_size < 2:
        raise ValueError("Workers/hidden must be nonnegative, epochs positive and batch size at least two.")
    if quantized_training and target_device.type != "cuda":
        raise ValueError("Quantized training benchmark profiles require CUDA.")
    if target_device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA profile requested but no accessible CUDA device is available.")
        torch.cuda.set_device(target_device)
        if dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("The selected CUDA device does not support bfloat16.")
        torch.cuda.synchronize(target_device)
        torch.cuda.reset_peak_memory_stats(target_device)
    repository = Path(__file__).resolve().parents[2]
    revision = (
        subprocess.run(["git", "rev-parse", "HEAD"], cwd=repository, capture_output=True, text=True, check=False).stdout.strip() or None
    )
    code_digest = hashlib.sha256()
    for source in sorted((repository / "mini_trainer").rglob("*.py")) + sorted(Path(__file__).parent.glob("*.py")):
        code_digest.update(str(source.relative_to(repository)).encode())
        code_digest.update(source.read_bytes())
    output = Path(output).absolute()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    if dataset == "synthetic":
        root = output / "data"
        manifest = generate(root, seed=seed)
        manifest_path = root / "manifest.json"
        spec_path = None
        size = 8
        model_type = "dev.benchmarks.models:ColorMean"
    else:
        if dataset not in ("mnist", "blair") or data_root is None:
            raise ValueError("Real profiles require dataset mnist/blair and --data-root.")
        root = Path(data_root).absolute()
        manifest, spec = prepare_real(root, output, name=dataset, seed=seed, class_spec=Path(class_spec) if class_spec else None)
        manifest_path = output / "dataset_manifest.json"
        spec_path = output / "class_spec.json"
        spec_path.write_text(json.dumps(spec, indent=2) + "\n")
        size = 28 if dataset == "mnist" else 64
        model_type = "dev.benchmarks.models:TinyConv"
    hierarchical = dataset == "blair"
    records = manifest["records"]
    train_records = [record for record in records if record["split"] != "test"]
    data_index = output / "train_index.json"
    data_index.write_text(
        json.dumps(
            {
                "path": [str(root / record["path"]) for record in train_records],
                "class": [record["targets"] if hierarchical else record["label"] for record in train_records],
                "split": [record["split"] for record in train_records],
            },
            indent=2,
        )
        + "\n"
    )
    started = time.perf_counter()
    train(
        input=str(root / "train"),
        class_spec=str(spec_path) if spec_path else None,
        output=str(output),
        name="training",
        device=device,
        dtype=dtype,
        epochs=epochs,
        size=size,
        seed=seed,
        builder=HierarchicalBenchmarkBuilder if hierarchical else NoAugmentationBuilder,
        ema=False,
        quantized_training=quantized_training,
        compile=compile,
        model_builder_kwargs={
            "model_type": model_type,
            "hidden": hidden if hidden else False,
            "normalized": hierarchical,
            "cls": HierarchicalClassifier if hierarchical else Classifier,
        },
        dataloader_builder_kwargs={
            "batch_size": batch_size,
            "num_workers": num_workers,
            "data_index": str(data_index),
            "cache": cache,
            "cache_workers": cache_workers,
        },
        optimizer_builder_kwargs={"optimizer_cls": MuonAuxAdamW, "lr": 0.1 if dataset == "synthetic" else 0.01, "weight_decay": 0.0},
        criterion_builder_kwargs={"label_smoothing": 0.0},
        regularizer_builder_kwargs={"strength": 0.0},
        lr_schedule_builder_kwargs={"warmup_epochs": 0.0},
        logger_builder_kwargs={"logger_cls": []},
    )
    if target_device.type == "cuda":
        torch.cuda.synchronize(target_device)
    elapsed = time.perf_counter() - started
    peak_memory = torch.cuda.max_memory_allocated(target_device) if target_device.type == "cuda" else None
    weights = output / "training/weights/last.pt"
    model, preprocess = Classifier.build(weights=str(weights), device=target_device, dtype=torch.float32)
    model.eval()
    quantization_recipe = getattr(model, "_quantized_training_recipe", None)
    if quantized_training and not quantization_recipe:
        raise RuntimeError("Quantized training recipe was not restored from the checkpoint.")
    test_records = [record for record in records if record["split"] == "test"]
    _, loader = get_inference_dataloader(
        images=[str(root / record["path"]) for record in test_records],
        resize_size=size,
        batch_size=batch_size,
        num_workers=0,
        device=target_device,
        dtype=torch.float32,
    )
    collected = []
    with torch.inference_mode(), torch.autocast(target_device.type, dtype=precision, enabled=precision != torch.float32):
        for batch in loader:
            logits = model(preprocess(batch.to(target_device)))
            if not isinstance(logits, (list, tuple)):
                logits = [logits]
            collected.append([value.float().cpu() for value in logits])
    scores_by_level = [torch.cat(values).numpy() for values in zip(*collected, strict=True)]
    labels_by_level = np.array([record["targets"] if hierarchical else [record["label"]] for record in test_records]).T
    accuracies = [float((scores.argmax(axis=1) == labels).mean()) for scores, labels in zip(scores_by_level, labels_by_level, strict=True)]
    if not all(np.isfinite(scores).all() for scores in scores_by_level):
        raise RuntimeError("Non-finite benchmark predictions.")
    accuracy = accuracies[0]
    arrays = {"scores": scores_by_level[0], "labels": labels_by_level[0], "paths": np.array([record["path"] for record in test_records])}
    arrays.update({f"scores_{level}": scores for level, scores in enumerate(scores_by_level)})
    arrays.update({f"labels_{level}": labels for level, labels in enumerate(labels_by_level)})
    np.savez(output / "predictions.npz", **arrays)
    result = {
        "schema_version": 1,
        "status": "passed" if dataset == "synthetic" and accuracy == 1.0 else "failed" if dataset == "synthetic" else "completed",
        "quality_gate": {"metric": "test_accuracy", "minimum": 1.0} if dataset == "synthetic" else None,
        "created_at": datetime.now(UTC).isoformat(),
        "git_revision": revision,
        "source_sha256": code_digest.hexdigest(),
        "lock_sha256": hashlib.sha256((repository / "uv.lock").read_bytes()).hexdigest(),
        "coverage": {
            "training": True,
            "checkpoint_reload": True,
            "inference": True,
            "hierarchical": hierarchical,
            "amp": precision != torch.float32,
            "cuda_cache": cache == "CUDA",
            "ema": False,
            "distributed": False,
            "quantization": bool(quantization_recipe),
            "augmentation": False,
            "onnx": False,
        },
        "dataset": dataset,
        "quantization_recipe": quantization_recipe,
        "compile": compile,
        "hidden": hidden,
        "batch_size": batch_size,
        "cache_workers": cache_workers,
        "parameter_bytes": sum(
            parameter.int_data.numel() + parameter.scale.numel() * parameter.scale.element_size()
            if getattr(parameter, "_is_quantized_training", False)
            else parameter.numel() * parameter.element_size()
            for parameter in model.parameters()
        ),
        "level_accuracies": accuracies,
        "split_counts": {split: sum(record["split"] == split for record in records) for split in ("train", "val", "test")},
        "seed": seed,
        "epochs": epochs,
        "oracle_accuracy": 1.0 if dataset == "synthetic" else None,
        "chance_accuracy": 1 / scores_by_level[0].shape[1],
        "test_accuracy": accuracy,
        "training_wall_seconds": elapsed,
        "training_wall_scope": "setup, training, validation, logging and checkpoints; includes first-use compilation/autotuning",
        "num_workers_requested": num_workers,
        "num_workers": 0 if cache == "CUDA" else num_workers,
        "cache": cache,
        "device": str(target_device),
        "dtype": dtype,
        "gpu_name": torch.cuda.get_device_name(target_device) if target_device.type == "cuda" else None,
        "peak_cuda_allocated_bytes": peak_memory,
        "amp_exercised": precision != torch.float32,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "torch_threads": torch.get_num_threads(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if target_device.type == "cuda" else None,
        "versions": {
            name: version(name)
            for name in (
                ("torch", "torchvision", "numpy", "mini_trainer", "torchao")
                if quantization_recipe
                else ("torch", "torchvision", "numpy", "mini_trainer")
            )
        },
        "dataset_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "checkpoint_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        "score_semantics": "model_eval_forward",
        "class_mapping": model.fc.metadata["cls2idx"],
        "test_used_for_training_or_selection": False,
    }
    (output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["synthetic", "mnist", "blair"], default="synthetic")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--class-spec", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--cache", choices=["NONE", "CPU", "RAM", "CUDA"], default="NONE")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cache-workers", type=int)
    parser.add_argument("--quantized-training", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--hidden", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--allow-nondeterministic",
        action="store_true",
        help="Allow backend operations without deterministic implementations; recorded in the report.",
    )
    args = parser.parse_args()
    if args.threads < 1 or args.epochs < 1:
        parser.error("threads and epochs must be positive")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(not args.allow_nondeterministic)
    if args.output.exists():
        parser.error("Output directory must be new.")
    try:
        result = run(
            args.output,
            args.seed,
            args.epochs,
            device=args.device,
            dtype=args.dtype,
            cache=args.cache,
            num_workers=args.num_workers,
            dataset=args.dataset,
            data_root=args.data_root,
            class_spec=args.class_spec,
            quantized_training=args.quantized_training,
            compile=args.compile,
            hidden=args.hidden,
            batch_size=args.batch_size,
            cache_workers=args.cache_workers,
        )
    except Exception as error:
        args.output.mkdir(parents=True, exist_ok=True)
        failure = {
            "schema_version": 1,
            "status": "failed",
            "dataset": args.dataset,
            "device": args.device,
            "dtype": args.dtype,
            "cache": args.cache,
            "seed": args.seed,
            "epochs": args.epochs,
            "quantized_training": args.quantized_training,
            "compile": args.compile,
            "hidden": args.hidden,
            "batch_size": args.batch_size,
            "cache_workers": args.cache_workers,
            "error": {"type": type(error).__name__, "message": str(error)},
        }
        (args.output / "report.json").write_text(json.dumps(failure, indent=2) + "\n")
        raise
    print(json.dumps(result, indent=2))
    if args.dataset == "synthetic" and result["test_accuracy"] < 1.0:
        raise SystemExit("Synthetic quality gate failed: expected oracle accuracy of 1.0.")


if __name__ == "__main__":
    main()
