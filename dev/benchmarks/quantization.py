"""Compare a recorded float checkpoint with PTQ and short QAT on training-only data."""

import hashlib
import json
import random
import subprocess
import time
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from mini_trainer.data import get_inference_dataloader
from mini_trainer.hierarchical.loss import MultiLevelWeightedCrossEntropyLoss
from mini_trainer.logging import MultiLogger
from mini_trainer.modeling import Classifier, EMATeacher
from mini_trainer.modeling.quantization import load_int8, prepare_int8
from mini_trainer.trainer import train_one_epoch


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(baseline, output, *, data_root=None, calibration_samples=256, qat_epochs=2, batch_size=32):
    baseline, output = Path(baseline), Path(output)
    if output.exists():
        raise FileExistsError(output)
    if batch_size < 2 or qat_epochs < 1 or calibration_samples < batch_size:
        raise ValueError("Require batch_size >= 2, qat_epochs >= 1 and calibration_samples >= batch_size.")
    baseline_report = json.loads((baseline / "report.json").read_text())
    dataset = baseline_report["dataset"]
    synthetic = dataset == "synthetic"
    root = baseline / "data" if synthetic else Path(data_root) if data_root else None
    if root is None:
        raise ValueError("Real datasets require --data-root.")
    manifest_path = baseline / "data/manifest.json" if synthetic else baseline / "dataset_manifest.json"
    weights = baseline / "training/weights/last.pt"
    for path, key in ((manifest_path, "dataset_manifest_sha256"), (weights, "checkpoint_sha256")):
        if digest(path) != baseline_report[key]:
            raise ValueError(f"Baseline provenance mismatch: {path}")
    manifest = json.loads(manifest_path.read_text())
    seed = baseline_report["seed"]
    torch.manual_seed(seed)
    train = [record for record in manifest["records"] if record["split"] == "train"]
    random.Random(seed).shuffle(train)
    # Keep full training batches: no duplicated or silently dropped examples.
    count = min(calibration_samples, len(train)) // batch_size * batch_size
    if count < batch_size:
        raise ValueError("Not enough training samples for a full calibration batch.")
    train = train[:count]
    test = [record for record in manifest["records"] if record["split"] == "test"]
    if not test:
        raise ValueError("A held-out test split is required.")
    for record in train + test:
        if digest(root / record["path"]) != record["sha256"]:
            raise ValueError(f"Dataset content changed: {record['path']}")
    model, preprocess = Classifier.build(weights=str(weights), device=torch.device("cpu"), dtype=torch.float32)
    model.eval()
    size = {"synthetic": 8, "mnist": 28, "blair": 64}[dataset]

    def read(records):
        _, loader = get_inference_dataloader(
            images=[str(root / record["path"]) for record in records],
            resize_size=size,
            batch_size=batch_size,
            num_workers=0,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        with torch.no_grad():
            return torch.cat([preprocess(images) for images in loader])

    images = read(train)
    targets = torch.tensor([record["targets"] if dataset == "blair" else record["label"] for record in train])
    example = images[:batch_size]
    output.mkdir(parents=True)
    provenance = {"split": "train", "seed": seed, "manifest_sha256": digest(manifest_path), "records": train}
    (output / "calibration.json").write_text(json.dumps(provenance, indent=2) + "\n")
    models = {"float": model}
    coverage = {}
    for mode in ("ptq", "qat"):
        prepared = prepare_int8(model, example, qat=mode == "qat")
        if mode == "ptq":
            with torch.no_grad():
                for batch in images.split(batch_size):
                    prepared(batch)
        else:
            loader = DataLoader(TensorDataset(images, targets), batch_size=batch_size, shuffle=False)
            optimizer = torch.optim.AdamW(prepared.parameters(), lr=1e-4, weight_decay=0)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100000)
            scaler = torch.amp.GradScaler("cpu", enabled=False)
            with torch.no_grad():
                original_output = model(example)
            criterion = (
                MultiLevelWeightedCrossEntropyLoss([v.shape[1] for v in original_output], torch.device("cpu"), torch.float32)
                if isinstance(original_output, list)
                else torch.nn.CrossEntropyLoss()
            )
            teacher = EMATeacher(enable=False, total_steps=qat_epochs * len(loader))
            logger = MultiLogger(loader, loader, epochs=qat_epochs, output=None, name="qat", logger_cls=[])
            for epoch in range(qat_epochs):
                train_one_epoch(prepared, teacher, criterion, optimizer, scaler, scheduler, loader, epoch, logger)
            torch.save(
                {
                    "model": prepared.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "epoch": qat_epochs - 1,
                },
                output / "qat_checkpoint.pt",
            )
        prepared.convert().save(
            output / mode,
            example,
            preprocessing={
                "source_checkpoint_sha256": digest(weights),
                "resize_size": size,
                "recipe": "Classifier.build checkpoint preprocessing; RGB uint8 input",
            },
            calibration=provenance,
        )
        models[mode], coverage[mode] = load_int8(output / mode).lower(example)
    # Test images enter the process only after conversion and all training finish.
    heldout = read(test)
    labels = np.array([record["targets"] if dataset == "blair" else [record["label"]] for record in test]).T
    results = {}
    for name, inference in models.items():
        collected = []
        with torch.no_grad():
            inference(example)  # Warm up outside timing.
            started = time.perf_counter()
            for batch in heldout.split(batch_size):
                n = len(batch)
                if n < batch_size:
                    batch = torch.cat([batch, batch[:1].expand(batch_size - n, *batch.shape[1:])])
                scores = inference(batch)
                collected.append([value[:n] for value in (scores if isinstance(scores, list) else [scores])])
            seconds = time.perf_counter() - started
        scores = [torch.cat(level).numpy() for level in zip(*collected, strict=True)]
        if not all(np.isfinite(level).all() for level in scores):
            raise ValueError(f"Nonfinite {name} predictions")
        accuracies = [float((level.argmax(1) == truth).mean()) for level, truth in zip(scores, labels, strict=True)]
        np.savez(
            output / f"{name}_predictions.npz",
            **{f"scores_{i}": v for i, v in enumerate(scores)},
            labels=labels,
            paths=np.array([record["path"] for record in test]),
        )
        results[name] = {
            "level_accuracies": accuracies,
            "test_inference_seconds": seconds,
            "artifact_bytes": weights.stat().st_size if name == "float" else (output / name / "model.pt2").stat().st_size,
        }
    repository = Path(__file__).resolve().parents[2]
    source_digest = hashlib.sha256()
    for source in sorted((repository / "mini_trainer").rglob("*.py")) + sorted(Path(__file__).parent.glob("*.py")):
        source_digest.update(str(source.relative_to(repository)).encode())
        source_digest.update(source.read_bytes())
    report = {
        "source_sha256": source_digest.hexdigest(),
        "lock_sha256": digest(repository / "uv.lock"),
        "schema_version": 1,
        "dataset": dataset,
        "seed": seed,
        "device": "cpu",
        "amp": False,
        "batch_size": batch_size,
        "threads": torch.get_num_threads(),
        "calibration_samples": count,
        "qat_epochs": qat_epochs,
        "qat_optimizer": {"name": "AdamW", "lr": 1e-4, "weight_decay": 0},
        "baseline_report_sha256": digest(baseline / "report.json"),
        "checkpoint_sha256": digest(weights),
        "calibration_sha256": digest(output / "calibration.json"),
        "results": results,
        "lowering": coverage,
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=repository).strip(),
        "status": "passed"
        if synthetic and all(r["level_accuracies"][0] == 1 for r in results.values())
        else "failed"
        if synthetic
        else "completed",
        "timing_scope": "single warmed full-test inference pass including padding and concatenation; not a speedup claim",
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--calibration-samples", type=int, default=256)
    parser.add_argument("--qat-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    torch.set_num_threads(1)
    report = run(**vars(args))
    print(json.dumps(report["results"], indent=2))
    if report["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
