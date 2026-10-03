"""Study builders using the production training loop and checkpoint format."""

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torchvision import transforms

from mini_trainer.builders import BaseBuilder
from mini_trainer.data.loader import get_dataset_dataloader
from mini_trainer.logging import MultiLogger, configure_loggers
from mini_trainer.modeling import Classifier, classification_module
from mini_trainer.train import main as train_main
from mini_trainer.training import EMLACrossEntropy, MuonAuxAdamW

from .data import digest, write_json


class FixedAdjustment(EMLACrossEntropy):
    """Same count handling and smoothing as EMLA, with a constant gate of one."""

    def forward(self, input, target):
        return nn.functional.cross_entropy(
            input + self.adjustments.to(input),
            target,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            label_smoothing=self.label_smoothing,
        )


class RandomStream:
    """Isolate torch CPU/device randomness without changing caller RNG state."""

    def __init__(self, function, seed, device="cpu"):
        self.function = function
        self.devices = [torch.device(device).index or 0] if torch.device(device).type == "cuda" else []
        with torch.random.fork_rng(devices=self.devices):
            torch.manual_seed(seed)
            self.cpu = torch.get_rng_state()
            self.cuda = [torch.cuda.get_rng_state(d) for d in self.devices]

    def __call__(self, *args, **kwargs):
        with torch.random.fork_rng(devices=self.devices):
            torch.set_rng_state(self.cpu)
            for device, state in zip(self.devices, self.cuda):
                torch.cuda.set_rng_state(state, device)
            result = self.function(*args, **kwargs)
            self.cpu = torch.get_rng_state()
            self.cuda = [torch.cuda.get_rng_state(d) for d in self.devices]
        return result


def tensor_hash(items):
    sha = hashlib.sha256()
    for name, value in sorted(items):
        if isinstance(value, torch.Tensor):
            sha.update(name.encode())
            sha.update(value.detach().cpu().contiguous().numpy().tobytes())
    return sha.hexdigest()


def initialize_head(head, seed):
    """Construct matching projection tensors independently of head treatment."""
    device = head.linear.weight.device
    with torch.random.fork_rng(devices=[device.index or 0] if device.type == "cuda" else []):
        torch.manual_seed(seed + 101)
        if head.hidden:
            head.hidden.reset_parameters()
        torch.manual_seed(seed + 102)
        layer = nn.Linear(head.preclassification_size, head.linear.out_features, device=device)
        head.linear = head._normalize_layer(layer, True) if head.normalized else layer


class StudyLogger(MultiLogger):
    def save(self, *args, **kwargs):
        if self.output_dir and self._epoch is not None:
            record = {"epoch": self._epoch, "phase": self._type}
            record.update({key: float(value) for key, value in self.summary().items() if value is not None and np.isfinite(value)})
            if torch.cuda.is_available():
                record["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            with (Path(self.output_dir) / "learning.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")


class StudyBuilder(BaseBuilder):
    # Configured only in a dedicated subprocess, never shared between runs.
    config = {}
    run = {}
    root = Path()
    attempt = Path()

    @classmethod
    def build_class_spec(cls, **kwargs):
        return json.loads((cls.root / "classes.json").read_text())

    @classmethod
    def build_model(cls, **kwargs):
        model, preprocess = super().build_model(**kwargs)
        head = classification_module(model)
        head_name = model._backbone_output_name
        pretrained = torch.load(cls.root / "pretrained.pt", map_location="cpu", weights_only=True)
        incompatible = model.load_state_dict({k: v for k, v in pretrained.items() if not k.startswith(head_name + ".")}, strict=False)
        if incompatible.unexpected_keys or any(not k.startswith(head_name + ".") for k in incompatible.missing_keys):
            raise ValueError("Pretrained backbone does not match study architecture")
        initialize_head(head, cls.run["seed"])
        write_json(
            cls.attempt / "initialization.json",
            {
                "backbone": tensor_hash((k, v) for k, v in model.state_dict().items() if not k.startswith(head_name + ".")),
                "backbone_parameters": tensor_hash((k, v) for k, v in model.named_parameters() if not k.startswith(head_name + ".")),
                "projection": tensor_hash(head.hidden.state_dict().items()) if head.hidden else None,
                "prototype": tensor_hash(head.linear.state_dict().items()),
            },
        )
        # Model construction consumes randomness differently between treatments.
        torch.manual_seed(cls.run["seed"] + 103)
        return model, preprocess

    @classmethod
    def frames(cls, split):
        frame = pd.read_parquet(cls.root / "samples.parquet", filters=[("split", "=", split)])
        if cls.run.get("qualification"):
            # Preserve full head/counts; this is an infrastructure fixture only.
            frame = frame.iloc[: max(2 * cls.config["batch_size"], 128)]
        return frame

    @classmethod
    def loader(cls, split, device):
        frame = cls.frames(split)
        metadata = {
            "path": [str(Path(cls.config["images"]) / key) for key in frame.sample_id],
            "class": frame.label.tolist(),
        }
        _, loaders = get_dataset_dataloader(
            metadata,
            resize_size=cls.config["size"],
            modes=(split,),
            batch_size=cls.config["batch_size"],
            num_workers=cls.config["workers"],
            device=device,
            cache=None,
        )
        loader = loaders[0]
        loader.generator = torch.Generator().manual_seed(cls.run["seed"] + 201)
        if split == "train":
            loader.batch_sampler.sampler.generator = torch.Generator().manual_seed(cls.run["seed"] + 202)
        return frame, loader

    @classmethod
    def build_dataloader(cls, device, **kwargs):
        frame, train = cls.loader("train", device)
        _, validation = cls.loader("validation", device)
        if len(train) == 0:
            raise ValueError("Training cohort smaller than one complete batch")
        return frame.label.tolist(), train, validation

    @classmethod
    def build_augmentation(cls, dtype):
        return transforms.Compose([RandomStream(super().build_augmentation(dtype), cls.run["seed"] + 301, cls.config["device"])])

    @classmethod
    def build_criterion(cls, device, **kwargs):
        spec = cls.build_class_spec()
        smoothing = 1 / spec["num_classes"]
        if cls.run["loss"] == "ce":
            return nn.CrossEntropyLoss(label_smoothing=smoothing)
        loss = EMLACrossEntropy if cls.run["loss"] == "emla" else FixedAdjustment
        return loss(spec["counts"], label_smoothing=smoothing, device=device)

    @classmethod
    def parameter_groups(cls, model, **kwargs):
        groups = super().parameter_groups(model, **kwargs)
        prototypes = {id(p) for p in classification_module(model).linear.parameters()}
        result = []
        for group in groups:
            for is_prototype in (False, True):
                params = [p for p in group["params"] if (id(p) in prototypes) == is_prototype]
                if params:
                    result.append({**group, "params": params, "weight_decay": 0 if is_prototype else group["weight_decay"]})
        return result

    @classmethod
    def build_optimizer(cls, model, **kwargs):
        optimizer = super().build_optimizer(model, **kwargs)
        names = {id(p): n for n, p in model.named_parameters()}
        write_json(
            cls.attempt / "parameter_groups.json",
            [
                {"name": g["name"], "lr": g["lr"], "weight_decay": g["weight_decay"], "parameters": [names[id(p)] for p in g["params"]]}
                for g in optimizer.param_groups
            ],
        )
        return optimizer

    @classmethod
    def build_regularizer(cls, **kwargs):
        function = RandomStream(super().build_regularizer(**kwargs), cls.run["seed"] + 401, cls.config["device"])

        def checked(model):
            value = function(model)
            if not torch.isfinite(torch.as_tensor(value)).all():
                raise FloatingPointError("Nonfinite prototype regularization")
            return value

        return checked

    @classmethod
    def build_logger(cls, **kwargs):
        return StudyLogger(**kwargs)


def train(root, attempt, config, run):
    if config["wandb"]:
        os.environ["WANDB_ENTITY"] = config["entity"]
    StudyBuilder.root, StudyBuilder.attempt = root, attempt
    StudyBuilder.config, StudyBuilder.run = config, run
    torch.set_num_threads(config["threads"])
    started = time.monotonic()
    train_main(
        input=str(root),
        output=str(attempt),
        name="model",
        epochs=run["epochs"],
        size=config["size"],
        device=config["device"],
        dtype=config["dtype"],
        builder=StudyBuilder,
        seed=run["seed"],
        ema=False,
        compile=False,
        model_builder_kwargs={
            "model_type": "efficientnet_v2_s",
            "model_args": {"pretrained": False},
            "hidden": run["hidden"],
            "normalized": run["normalized"],
            "droprate": 0.1,
            "fine_tune": False,
            "skip_spherical_init": True,
        },
        dataloader_builder_kwargs={"batch_size": config["batch_size"]},
        optimizer_builder_kwargs={
            "optimizer_cls": MuonAuxAdamW if run["optimizer"] == "muon" else torch.optim.AdamW,
            "lr": run["lr"],
            "backbone_lr": run["lr"] / 3,
            "weight_decay": run["weight_decay"],
        },
        criterion_builder_kwargs={},
        regularizer_builder_kwargs={"strength": 0.1 if run["regularization"] else 0},
        lr_schedule_builder_kwargs={"warmup_epochs": 1, "pretrained_backbone": True},
        logger_builder_kwargs={
            "verbose": True,
            "logger_cls": configure_loggers(use_wandb=config["wandb"]),
            "logger_cls_extra_kwargs": [{}, {"project": config["project"], "run_name": run["id"]}],
        },
    )
    weights = attempt / "model/weights/last.pt"
    elapsed = time.monotonic() - started
    curves = [json.loads(line) for line in (attempt / "model/logs/learning.jsonl").read_text().splitlines()]
    samples_per_epoch = len(StudyBuilder.frames("train")) // config["batch_size"] * config["batch_size"]
    write_json(
        attempt / "train.json",
        {
            "weights_sha256": digest(weights),
            "wall_seconds": elapsed,
            "images_per_second_including_validation_and_logging": samples_per_epoch * run["epochs"] / elapsed,
            "peak_allocated_bytes": max((row.get("peak_allocated_bytes", 0) for row in curves), default=0),
        },
    )


def metrics(logits, targets, counts):
    logits = torch.as_tensor(logits, dtype=torch.float64)
    targets = torch.tensor(np.asarray(targets), dtype=torch.long)
    if not torch.isfinite(logits).all() or not len(targets):
        raise ValueError("Empty evaluation or nonfinite predictions")
    probabilities = logits.softmax(-1)
    predicted = probabilities.argmax(-1)
    support = torch.bincount(targets, minlength=len(counts))
    hits = torch.bincount(targets[predicted == targets], minlength=len(counts))
    recalls = hits.double() / support.clamp_min(1)
    result = {
        "accuracy": float((predicted == targets).double().mean()),
        "macro_recall": float(recalls[support > 0].mean()),
        "nll": float(nn.functional.cross_entropy(logits, targets)),
        "brier": float((probabilities.square().sum(-1) - 2 * probabilities[torch.arange(len(targets)), targets] + 1).mean()),
        "support": support.tolist(),
        "recall": [float(v) if s else None for v, s in zip(recalls, support)],
    }
    for name, indices in zip(["tail", "mid", "head"], np.array_split(np.argsort(counts, kind="stable"), 3)):
        indices = indices[support[indices].numpy() > 0]
        result[name + "_recall"] = float(recalls[indices].mean()) if len(indices) else None
    return result


def evaluate(root, attempt, config, run):
    StudyBuilder.root, StudyBuilder.attempt = root, attempt
    StudyBuilder.config, StudyBuilder.run = config, run
    split = "validation" if run.get("tuning") or run.get("qualification") else "test"
    frame, loader = StudyBuilder.loader(split, config["device"])
    weights = attempt / "model/weights/last.pt"
    record = json.loads((attempt / "train.json").read_text())
    if digest(weights) != record["weights_sha256"]:
        raise ValueError("Completed training weights changed")
    model, preprocess = Classifier.build(
        weights=str(weights), device=config["device"], model_args={"pretrained": False}, skip_spherical_init=True
    )
    model.eval()
    backbone_hash = tensor_hash((k, v) for k, v in model.named_parameters() if not k.startswith(model._backbone_output_name + "."))
    initial = json.loads((attempt / "initialization.json").read_text())
    backbone_changed = backbone_hash != initial["backbone_parameters"]
    if run.get("qualification") and not backbone_changed:
        raise ValueError("Qualification did not update backbone parameters after warmup")
    predictions = []
    started = time.monotonic()
    with torch.inference_mode():
        for images, _ in loader:
            predictions.append(model(preprocess(images.to(config["device"]))).float().cpu())
    logits = torch.cat(predictions).numpy()
    result = metrics(logits, frame.label.to_numpy(), StudyBuilder.build_class_spec()["counts"])
    np.savez_compressed(
        attempt / "predictions.npz", logits=logits, target=frame.label.to_numpy(), sample_id=frame.sample_id.to_numpy(dtype=str)
    )
    result.update(
        {
            "split": split,
            "backbone_parameters_changed": backbone_changed,
            "wall_seconds": time.monotonic() - started,
            "weights_sha256": record["weights_sha256"],
            "predictions_sha256": digest(attempt / "predictions.npz"),
        }
    )
    write_json(attempt / "evaluation.json", result)
