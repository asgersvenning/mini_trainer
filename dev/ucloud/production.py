"""Generate a production CLI config from verified full-taxonomy qualification artifacts."""

import argparse
import hashlib
import json
from pathlib import Path

import yaml

STATISTICS = [
    "acc1",
    "acc5",
    "loss",
    "lr",
    "item/s",
    "mem",
    "step",
    "time",
    "eta",
    "epoch",
    "type",
    "acc1/lvl1",
    "acc5/lvl1",
    "acc1/lvl2",
    "acc5/lvl2",
    "loss/lvl0",
    "loss/lvl1",
    "loss/lvl2",
]


def generate(root, destination, output, name, batch, workers, epochs, project):
    root, destination, output = Path(root).resolve(), Path(destination).resolve(), Path(output).resolve()
    if min(batch, epochs) < 1 or workers < 0:
        raise ValueError("Positive batch/epochs and nonnegative workers required")
    if not name or Path(name).name != name or name in (".", ".."):
        raise ValueError("Run name must be a single directory name")
    config = json.loads((root / "comparison.json").read_text())
    if config.get("mode") != "scaling" or not config.get("full_taxonomy") or config["gpus"] != 4:
        raise ValueError("Require full-taxonomy four-GPU scaling preparation")
    manifest = json.loads((root / "prepared.json").read_text())
    seed = config["seeds"][0]
    for filename in ("class_spec.json", "dataset.json", f"initial_seed{seed}.pt"):
        with (root / filename).open("rb") as handle:
            actual = hashlib.file_digest(handle, "sha256").hexdigest()
        if actual != manifest[filename]:
            raise ValueError(f"Prepared artifact changed: {filename}")
    if (output / name).exists():
        raise ValueError("Production output already exists; use a new run name or reviewed resume procedure")
    settings = {
        "input": config["parquet"],
        "output": str(output),
        "name": name,
        "class_spec": str(root / "class_spec.json"),
        "epochs": epochs,
        "size": config["size"],
        "seed": seed,
        "device": "cuda",
        "dtype": "float16",
        "ema": False,
        "compile": True,
        "compile_optimizer": False,
        "quantized_training": False,
        "model_builder_kwargs": {
            "model_type": "efficientnet_v2_s",
            "weights": str(root / f"initial_seed{seed}.pt"),
            "model_args": {"pretrained": False},
            "hidden": True,
            "normalized": True,
            "droprate": 0.1,
            "fine_tune": False,
        },
        "dataloader_builder_kwargs": {
            "batch_size": batch,
            "num_workers": workers,
            "cache": None,
            "splits": ["train", "val"],
            "resample": False,
            "cuda_prefetch": False,
        },
        "optimizer_builder_kwargs": {"optimizer_cls": "mini_trainer.training.MuonAuxAdamW", "lr": 0.001, "weight_decay": 0.01},
        "criterion_builder_kwargs": {"weighted": True, "weights": [1, 1, 1]},
        "regularizer_builder_kwargs": {"strength": 0.1},
        "lr_schedule_builder_kwargs": {"warmup_epochs": 0.25},
        "logger_builder_kwargs": {
            "verbose": True,
            "statistics": STATISTICS,
            "logger_cls_extra_kwargs": [
                {},
                {"project": project, "run_name": name, "run_id": hashlib.sha256(str(output / name).encode()).hexdigest()[:16]},
            ],
        },
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x") as handle:
        yaml.safe_dump(settings, handle, sort_keys=False)
    return settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("qualification", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="Persistent production output parent")
    parser.add_argument("--name", required=True)
    parser.add_argument("--batch", type=int, required=True, help="Per GPU")
    parser.add_argument("--workers", type=int, required=True, help="Per GPU")
    parser.add_argument("--epochs", type=int, required=True, help="Reviewed full-dataset scheduler horizon")
    parser.add_argument("--project", default="mini-trainer-production")
    args = parser.parse_args()
    config = generate(args.qualification, args.destination, args.output, args.name, args.batch, args.workers, args.epochs, args.project)
    print(f"Wrote {args.destination}; global batch={4 * args.batch}, epochs={args.epochs}")
    print(f"Full source: {config['input']}; no qualification data_index is used")
    print("Launch with the pinned environment's mt_htrain --config FILE --wandb --compile under four-rank torchrun")


if __name__ == "__main__":
    main()
