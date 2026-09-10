"""Generate bounded DDP trials and summarize per-rank throughput/loading evidence."""

import argparse
import json
import math
import statistics
from pathlib import Path

from compare import validate


def trial(base, destination, output, batch, epochs, checkpoint=None, resume_epoch=None, workers=None, storage=False):
    config = json.loads(Path(base).read_text())
    original_output = config["output"]
    config.pop("checkpoint", None)
    config.pop("resume_epoch", None)
    config.update(
        global_batch_size=batch * config["gpus"], epochs=epochs, output=str(Path(output).resolve()), reuse_preparation=original_output
    )
    if workers is not None:
        config["num_workers_per_rank"] = workers
    validate(config)
    if storage:
        if checkpoint:
            raise ValueError("Storage selection cannot be combined with checkpoint restoration")
        source = Path(original_output)
        manifest = json.loads((source / "prepared.json").read_text())
        config.pop("reuse_preparation")
        config.update(
            exclude_qualification=str(source / "qualification.parquet"),
            exclude_qualification_sha256=manifest["qualification.parquet"],
            epochs=1,
            timeout_seconds=600,
            budget_seconds=1800,
        )
        config["qualification"] = dict(config["qualification"], train=262144, validation=8192, test=128)
    elif not checkpoint and config.get("qualification"):
        steps = config["qualification"]["train"] // config["global_batch_size"]
        if steps < 1:
            raise ValueError("Need at least one complete global batch")
        config["epochs"] = max(epochs, 1 + math.ceil(100 / steps))
    if checkpoint:
        config.update(checkpoint=str(Path(checkpoint).resolve()), resume_epoch=resume_epoch)
    config = validate(config)
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    with Path(destination).open("x") as handle:
        json.dump(config, handle, indent=2)
        handle.write("\n")
    return config


def report(root):
    root = Path(root)
    config = json.loads((root / "comparison.json").read_text())
    environment = json.loads((root / "environment-quant.json").read_text())
    results = []
    for result in sorted(root.glob("runs/*/result.json")):
        row = json.loads(result.read_text())
        ranks = []
        for rank in range(config["gpus"]):
            path = result.parent / f"phases-rank{rank}.jsonl"
            ranks.append([json.loads(line) for line in path.read_text().splitlines()] if path.is_file() else [])
        epochs = sorted(set(p["epoch"] for rank in ranks for p in rank if p["phase"] == "train"))
        measurements = []
        for epoch in epochs:
            phases = [next((p for p in rank if p["phase"] == "train" and p["epoch"] == epoch), None) for rank in ranks]
            if any(p is None for p in phases):
                continue
            seconds = max(p["seconds"] for p in phases)
            measurements.append(
                {
                    "epoch": epoch,
                    "images_per_second": sum(p["samples"] for p in phases) / seconds,
                    "seconds": seconds,
                    "rank_time_ratio": seconds / min(p["seconds"] for p in phases),
                    "steps": min(p.get("steps", 0) for p in phases),
                    "windows_by_rank": [p.get("windows", []) for p in phases],
                    "max_loader_wait_seconds": max(p["loader_wait_seconds"] for p in phases),
                }
            )
        warm = measurements[1:]
        peaks = [max((p["max_memory_reserved"] for p in rank), default=0) for rank in ranks]
        row.update(
            warm_steps=sum(p["steps"] for p in warm),
            sufficient_warm_steps=sum(p["steps"] for p in warm) >= 100,
            performance_eligible=(row.get("status") == "completed" and all(ranks) and sum(p["steps"] for p in warm) >= 100),
            phases_by_rank=ranks,
            component_timings_by_rank=[
                [json.loads(line) for line in path.read_text().splitlines()]
                for path in sorted(result.parent.glob("components-rank*.jsonl"))
            ],
            peak_allocated_bytes=max((p.get("max_memory_allocated", 0) for rank in ranks for p in rank), default=0),
            per_gpu_batch=config["global_batch_size"] // config["gpus"],
            epochs=measurements,
            warm_images_per_second=statistics.median(p["images_per_second"] for p in warm) if warm else None,
            peak_reserved_fraction=max(p / total for p, total in zip(peaks, environment["gpu_memory_bytes"], strict=True)),
            all_ranks_recorded=all(ranks),
            restore_checks=len(list(result.parent.glob("restore-rank*.json"))),
        )
        results.append(row)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    create = commands.add_parser("trial")
    create.add_argument("base")
    create.add_argument("destination")
    create.add_argument("--output", required=True)
    create.add_argument("--batch", type=int, required=True, help="Per GPU batch")
    create.add_argument("--epochs", type=int, default=3)
    create.add_argument("--workers", type=int, help="Loader workers per rank")
    create.add_argument("--storage", action="store_true", help="Separate campaign with a larger disjoint selection")
    create.add_argument("--checkpoint")
    create.add_argument("--resume-epoch", type=int)
    summarize = commands.add_parser("report")
    summarize.add_argument("root")
    args = parser.parse_args()
    if args.action == "trial":
        trial(
            args.base,
            args.destination,
            args.output,
            args.batch,
            args.epochs,
            args.checkpoint,
            args.resume_epoch,
            args.workers,
            args.storage,
        )
    else:
        print(json.dumps(report(args.root), indent=2))


if __name__ == "__main__":
    main()
