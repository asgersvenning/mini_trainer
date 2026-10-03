"""Short LR range/hold probes using the production loop; not ablation results."""

import argparse
import json
import math
from pathlib import Path

import torch

from . import study, training
from .data import digest, write_json
from .operational import ProfileBuilder


class ProbeStop(RuntimeError):
    """A recorded operational divergence criterion ended the probe."""


def lr_factor(step, warmup, lower, upper, backbone=False, hold=False):
    if step < warmup:
        return 0.0 if backbone else ((upper if hold else 0.0003) / upper) * (1e-4 + (1 - 1e-4) * step / warmup)
    if hold:
        return 1.0
    position = min((step - warmup) / max(warmup - 1, 1), 1)
    return lower / upper * (upper / lower) ** position


class ProbeScaler(torch.amp.GradScaler):
    def unscale_(self, optimizer):
        result = super().unscale_(optimizer)
        norms = {}
        for name, params in RangeBuilder.gradient_groups.items():
            values = [p.grad.detach().float().norm() for p in params if p.grad is not None]
            norm = float(torch.stack(values).norm()) if values else 0.0
            norms[name] = norm if math.isfinite(norm) else None
        RangeBuilder.record = {
            "group_lrs": [float(g["lr"]) for g in optimizer.param_groups],
            "backbone_active": any("head" not in g.get("name", "head") and g["lr"] > 0 for g in optimizer.param_groups),
            "gradient_norms_before_clipping": norms,
            "scale_before": self.get_scale(),
        }
        return result

    def update(self, new_scale=None):
        super().update(new_scale)
        RangeBuilder.record["scale_after"] = self.get_scale()
        RangeBuilder.record["amp_skipped"] = self.get_scale() < RangeBuilder.record["scale_before"]


class RangeLogger(training.StudyLogger):
    def consume(self, **kwargs):
        super().consume(**kwargs)
        if self._type != "train":
            return
        loss = kwargs["loss"]
        loss = float((sum(loss) if isinstance(loss, list) else loss).detach())
        row = {**RangeBuilder.record, "epoch": self._epoch, "batch": kwargs["index"], "loss": loss}
        with (RangeBuilder.attempt / "lr-curve.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        RangeBuilder.skips = RangeBuilder.skips + 1 if row["amp_skipped"] else 0
        if RangeBuilder.skips >= 5:
            raise ProbeStop("Five consecutive AMP-skipped updates")
        if row["backbone_active"]:
            RangeBuilder.smoothed = loss if RangeBuilder.smoothed is None else 0.9 * RangeBuilder.smoothed + 0.1 * loss
            RangeBuilder.best = min(RangeBuilder.best, RangeBuilder.smoothed)
            if kwargs["index"] >= 20 and RangeBuilder.smoothed > 4 * RangeBuilder.best:
                raise ProbeStop("Smoothed loss exceeded four times its best full-model value")


class RangeBuilder(ProfileBuilder):
    gradient_groups = {}
    record = {}
    skips = 0
    smoothed = None
    best = math.inf
    lower = 1e-5
    upper = 1.0
    hold = False

    @classmethod
    def build_optimizer(cls, model, **kwargs):
        optimizer = super().build_optimizer(model, **kwargs)
        head = training.classification_module(model)
        projection = {id(p) for p in head.hidden.parameters()} if head.hidden else set()
        classifier = {id(p) for p in head.parameters()} - projection
        cls.gradient_groups = {
            "projection": [p for p in model.parameters() if id(p) in projection],
            "classifier": [p for p in model.parameters() if id(p) in classifier],
            "backbone": [p for p in model.parameters() if id(p) not in projection | classifier],
        }
        return optimizer

    @classmethod
    def build_lr_scheduler(cls, optimizer, steps_per_epoch, **kwargs):
        # Keep LR changes gated by successful updates in the production trainer.
        functions = [
            lambda step, backbone="head" not in group.get("name", "head"): lr_factor(
                step, steps_per_epoch, cls.lower, cls.upper, backbone, cls.hold
            )
            for group in optimizer.param_groups
        ]
        return torch.optim.lr_scheduler.LambdaLR(optimizer, functions)

    @classmethod
    def build_scaler(cls, device, **kwargs):
        return ProbeScaler(device=torch.device(device).type, init_scale=2**14, growth_interval=100, **kwargs)

    @classmethod
    def build_logger(cls, **kwargs):
        return RangeLogger(**kwargs)


def probe(root, output, optimizer, batches=128, lower=1e-5, upper=1.0, hold=False):
    config = dict(study.verify(root))
    if batches < 32 or not 0 < lower <= upper:
        raise ValueError("Use at least 32 batches and 0 < lower <= upper")
    output.mkdir(parents=True, exist_ok=False)
    run = {
        **study.FULL,
        "optimizer": optimizer,
        "id": f"lr_{'hold' if hold else 'range'}_{optimizer}_{upper}",
        "seed": 41,
        "epochs": 2,
        "lr": upper,
        "weight_decay": 0.001,
    }
    RangeBuilder.train_samples = batches * config["batch_size"]
    RangeBuilder.lower, RangeBuilder.upper, RangeBuilder.hold = lower, upper, hold
    RangeBuilder.skips, RangeBuilder.smoothed, RangeBuilder.best = 0, None, math.inf
    RangeBuilder.root, RangeBuilder.config, RangeBuilder.run = root, config, run
    frame = RangeBuilder.frames("train")
    frame.to_parquet(output / "sample.parquet", index=False)
    write_json(output / "run.json", run)
    write_json(output / "resolved.json", config)
    write_json(
        output / "probe.json",
        {
            "prepared_sha256": digest(root / "prepared.json"),
            "sample_sha256": digest(output / "sample.parquet"),
            "lower": lower,
            "upper": upper,
            "hold": hold,
            "batches_per_epoch": len(frame) // config["batch_size"],
        },
    )
    original = training.StudyBuilder
    training.StudyBuilder = RangeBuilder
    reason = "Hold completed" if hold else "Probe budget completed without a detected boundary; inspect the actual LR coverage"
    try:
        training.train(root, output, config, run)
    except ProbeStop as error:
        reason = str(error)
    except (FloatingPointError, RuntimeError) as error:
        # Unrelated failures (including OOM/data errors) are never LR evidence.
        if str(error) not in ["Nonfinite prototype regularization", "Interrupted training due to persistent nan's detected in the loss."]:
            raise
        reason = str(error)
    finally:
        training.StudyBuilder = original
    rows = (
        [json.loads(line) for line in (output / "lr-curve.jsonl").read_text().splitlines()] if (output / "lr-curve.jsonl").exists() else []
    )
    write_json(
        output / "result.json",
        {
            "reason": reason,
            "recorded_batches": len(rows),
            "last_recorded_step": rows[-1] if rows else None,
            "limits": (
                "Short sampled probe with gradient clipping and AMP; ramp history is confounded with LR. "
                "Confirm candidates in fresh fixed-LR holds. No optimal LR or universal stability claim."
            ),
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--optimizer", choices=["muon", "adamw"], required=True)
    parser.add_argument("--batches", type=int, default=128)
    parser.add_argument("--lower", type=float, default=1e-5)
    parser.add_argument("--upper", type=float, default=1.0)
    parser.add_argument("--hold", action="store_true")
    args = parser.parse_args()
    probe(args.root.resolve(), args.output.resolve(), args.optimizer, args.batches, args.lower, args.upper, args.hold)


if __name__ == "__main__":
    main()
