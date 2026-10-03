"""Bounded random-cohort timing using the real study trainer, outside the run matrix."""

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch

from . import study, training
from .data import digest, write_json


def sample_frame(frame, limit, seed):
    # Sort before sampling so metadata row order does not select different images.
    return frame.sort_values("sample_id").sample(n=min(limit, len(frame)), random_state=seed).reset_index(drop=True)


class ProfileLogger(training.StudyLogger):
    def update(self, epoch, type):
        super().update(epoch, type)
        self.previous = None

    def consume(self, **kwargs):
        super().consume(**kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        now = time.monotonic()
        if self.previous is not None:
            with (Path(self.output_dir) / "batch-timing.jsonl").open("a") as stream:
                stream.write(
                    json.dumps(
                        {
                            "epoch": self._epoch,
                            "phase": self._type,
                            "index": kwargs["index"],
                            "images": len(kwargs["batch"]),
                            "seconds": now - self.previous,
                        }
                    )
                    + "\n"
                )
        self.previous = now


class ProfileBuilder(training.StudyBuilder):
    train_samples = 4096

    @classmethod
    def frames(cls, split):
        frame = pd.read_parquet(cls.root / "samples.parquet", filters=[("split", "=", split)])
        limit = cls.train_samples if split == "train" else 4 * cls.config["batch_size"]
        return sample_frame(frame, limit, cls.run["seed"])

    @classmethod
    def build_logger(cls, **kwargs):
        return ProfileLogger(**kwargs)


def profile(root, output, samples):
    config = study.verify(root)
    if (root / "qualified.json").exists():
        config["batch_size"] = json.loads((root / "qualified.json").read_text())["batch_size"]
    if samples < 12 * config["batch_size"]:
        raise ValueError("Use at least 12 training batches for a timing window after warmup")
    output.mkdir(parents=True, exist_ok=False)
    run = {**study.FULL, "id": "operational_profile", "seed": 39, "epochs": 2, "qualification": True, "lr": 0.001, "weight_decay": 0.01}
    write_json(output / "run.json", run)
    write_json(output / "resolved.json", config)
    ProfileBuilder.train_samples = samples
    # This helper is a dedicated process; production study workers keep their builder.
    training.StudyBuilder = ProfileBuilder
    ProfileBuilder.root, ProfileBuilder.config, ProfileBuilder.run = root, config, run
    train_frame = ProfileBuilder.frames("train")
    train_frame.to_parquet(output / "sample.parquet", index=False)
    training.train(root, output, config, run)
    records = [json.loads(line) for line in (output / "model/logs/batch-timing.jsonl").read_text().splitlines()]
    # Epoch zero is head-only LR warmup. Measure trainable-backbone epoch one,
    # excluding its first four batches (worker startup and prefetch transients).
    measured = [r for r in records if r["epoch"] == 1 and r["phase"] == "train" and r["index"] >= 4]
    if len(measured) < 8:
        raise RuntimeError("Insufficient completed training batches for timing")
    seconds = sum(r["seconds"] for r in measured)
    rate = sum(r["images"] for r in measured) / seconds
    _, loader = ProfileBuilder.loader("train", torch.device(config["device"]))
    started = time.monotonic()
    images = 0
    for batch, _ in loader:
        images += len(batch)
    if config["device"] == "cuda":
        torch.cuda.synchronize()
    loader_seconds = time.monotonic() - started
    training.evaluate(root, output, config, run)
    training_record = json.loads((output / "train.json").read_text())
    if config["device"] == "cuda" and training_record["peak_allocated_bytes"] <= 0:
        raise RuntimeError("GPU peak-memory evidence is missing")
    total_train = len(pd.read_parquet(root / "samples.parquet", filters=[("split", "=", "train")]))
    validation = [r for r in records if r["epoch"] == 1 and r["phase"] == "eval"]
    validation_rate = sum(r["images"] for r in validation) / sum(r["seconds"] for r in validation)
    total_validation = len(pd.read_parquet(root / "samples.parquet", filters=[("split", "=", "validation")]))
    epoch_seconds = total_train // config["batch_size"] * config["batch_size"] / rate + total_validation / validation_rate
    write_json(
        output / "profile.json",
        {
            "prepared_sha256": digest(root / "prepared.json"),
            "sample_sha256": digest(output / "sample.parquet"),
            "sample_images": len(train_frame),
            "sample_species": train_frame.label.nunique(),
            "training_batches_measured": len(measured),
            "training_images_per_second": rate,
            "training_batch_seconds": [r["seconds"] for r in measured],
            "projected_train_seconds_per_epoch": total_train // config["batch_size"] * config["batch_size"] / rate,
            "validation_images_per_second": validation_rate,
            "projected_train_validation_seconds_per_epoch": epoch_seconds,
            "allocation_planning_scenarios_hours": {
                "steady_state_30_epochs": 30 * epoch_seconds / 3600,
                "half_throughput_plus_pilot_wall_per_epoch": 30 * (2 * epoch_seconds + training_record["wall_seconds"]) / 3600,
            },
            "loader_only_images_per_second": images / loader_seconds,
            "peak_allocated_bytes": training_record["peak_allocated_bytes"],
            "device": torch.cuda.get_device_name() if config["device"] == "cuda" else "cpu",
            "limits": (
                "Uniform image sample, not class-balanced. Training reuses images in epoch two; "
                "loader-only pass follows training and may hit filesystem caches. Batch intervals include "
                "loading, augmentation, compute and logging with CUDA synchronization. Train-only projection excludes "
                "validation, evaluation, checkpointing and startup; not a full-campaign runtime estimate. "
                "One full-recipe optimizer only. Allocation scenarios are sensitivity calculations, not confidence bounds; "
                "the slower scenario halves throughput and adds the whole pilot wall time each epoch."
            ),
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Prepared campaign at the current source/environment revision")
    parser.add_argument("output", type=Path, help="Fresh output directory, separate from campaign runs")
    parser.add_argument("--samples", type=int, default=4096)
    args = parser.parse_args()
    profile(args.root.resolve(), args.output.resolve(), args.samples)


if __name__ == "__main__":
    main()
