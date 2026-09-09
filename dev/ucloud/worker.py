"""Branch-neutral worker, executed with the selected installed-package interpreter."""

import argparse
import functools
import importlib.metadata
import json
import os
import platform
import random
import shutil
import tempfile
import time
from collections import Counter
from pathlib import Path

from compare import digest, plan, write_json


def preflight(config, branch, *, verify=False):
    import torch

    import mini_trainer

    distribution = importlib.metadata.distribution("mini_trainer")
    direct = json.loads(distribution.read_text("direct_url.json") or "{}")
    commit = direct.get("vcs_info", {}).get("commit_id")
    if commit != config["environments"][branch]["commit"]:
        raise ValueError("Install from the pinned Git URL (non-editable); installed commit differs or has no VCS provenance")
    compiler = shutil.which(os.environ.get("CXX", "c++"))
    if compiler is None:
        raise RuntimeError("A C++ compiler is required, including for existing compiled augmentation kernels")
    if not torch.cuda.is_available() or torch.cuda.device_count() < config["gpus"]:
        raise RuntimeError(f"Need {config['gpus']} visible CUDA GPUs")
    if not verify:
        for index in range(config["gpus"]):
            with torch.cuda.device(index):
                torch.ones(1, device=f"cuda:{index}").add_(1)
                torch.cuda.synchronize()
    if "quant_int8" in config["variants"]:
        importlib.metadata.version("torchao")
    record = {
        "commit": commit,
        "package": mini_trainer.__file__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "compiler": compiler,
        "cuda": torch.version.cuda,
        "gpus": [torch.cuda.get_device_name(i) for i in range(config["gpus"])],
        "dependencies": {
            d.metadata["Name"].lower(): d.version
            for d in importlib.metadata.distributions()
            if d.metadata["Name"].lower().replace("-", "_") != "mini_trainer"
        },
    }
    destination = Path(config["output"]) / f"environment-{branch}.json"
    if verify:
        if record != json.loads(destination.read_text()):
            raise ValueError("Installed environment or GPU allocation changed since preparation")
    else:
        write_json(destination, record)


def prepare(config):
    import numpy as np
    import torch

    from mini_trainer.hierarchical.integration import HierarchicalBuilder
    from mini_trainer.integrations.parquet import get_metadata_from_parquet

    output = Path(config["output"])
    spec = HierarchicalBuilder.build_class_spec(dir=config["parquet"], levels=3)
    spec["resize_size"] = config["size"]
    write_json(output / "class_spec.json", spec)
    data = get_metadata_from_parquet(config["parquet"], cls2idx=spec["cls2idx"])
    counts = Counter(data["split"])
    if not all(counts[split] for split in ("train", "validation", "test")):
        raise ValueError(f"Require nonempty existing train/validation/test splits, got {counts}")
    if len(set(data["path"])) != len(data["path"]):
        raise ValueError("Duplicate image paths; resolve duplicates/split leakage before comparing")
    for path, labels in zip(data["path"], data["class"], strict=True):
        if not Path(path).is_file():
            raise FileNotFoundError(f"Expected Parquet-adjacent images/<speciesKey>/<filename>: {path}")
        if any(value is None for value in labels):
            raise ValueError(f"Unmapped taxonomy: {path}")
    per_rank = config["global_batch_size"] // config["gpus"]
    if counts["train"] // config["gpus"] < per_rank:
        raise ValueError("Not enough training images for one global batch")
    write_json(output / "data_index.json", data)
    smoothing = 1 / spec["num_classes"][0]
    write_json(
        output / "dataset.json",
        {
            "parquet_sha256": digest(config["parquet"]),
            "counts": counts,
            "num_classes": spec["num_classes"],
            "label_smoothing_per_level": [1 - (1 - smoothing) ** (1 / (i + 1)) for i in range(3)],
            "image_integrity": "All paths checked; image bytes are not hashed or decoded during preparation. Keep mount immutable.",
            "test_used_for_training_or_selection": False,
        },
    )
    for seed in config["seeds"]:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        model, preprocess = HierarchicalBuilder.build_model(
            **spec,
            model_type="efficientnet_v2_s",
            device=torch.device("cpu"),
            dtype=torch.float32,
            hidden=True,
            normalized=True,
            droprate=0.1,
            fine_tune=False,
        )
        torch.save(model.state_dict(), output / f"initial_seed{seed}.pt")
        (output / "preprocessing.txt").write_text(repr(preprocess) + "\n")
        del model


def instrument(output):
    """Record synchronized phase times per rank without changing optimizer logic."""
    import torch

    import mini_trainer.trainer as trainer
    from mini_trainer.logging import MultiLogger

    # The normal logger resets peak stats within a phase. Disable only that
    # measurement reset so phase-level CUDA peaks include all steps.
    MultiLogger._reset_cuda_memory_stats = staticmethod(lambda: None)
    rank = int(os.environ.get("RANK", "0"))

    def wrap(function, phase):
        @functools.wraps(function)
        def measured(*args, **kwargs):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            value = function(*args, **kwargs)
            torch.cuda.synchronize()
            record = {
                "phase": phase,
                "seconds": time.monotonic() - started,
                "max_memory_allocated": torch.cuda.max_memory_allocated(),
                "max_memory_reserved": torch.cuda.max_memory_reserved(),
            }
            if phase == "validation":
                record["selection_metric"] = float(value)
            with (output / f"phases-rank{rank}.jsonl").open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            return value

        return measured

    trainer.train_one_epoch = wrap(trainer.train_one_epoch, "train")
    trainer.evaluate = wrap(trainer.evaluate, "validation")


def train(config, name):
    import torch

    from mini_trainer.hierarchical.integration import HierarchicalBuilder
    from mini_trainer.train import main
    from mini_trainer.training import MuonAuxAdamW

    run = next(run for run in plan(config) if run["name"] == name)
    preflight(config, run["branch"], verify=True)
    root = Path(config["output"])
    directory = root / "runs" / name
    # Isolate compiler caches by run and rank; compilation cost is included.
    rank = os.environ.get("RANK", "0")
    cache_root = os.environ.get("MT_COMPILER_CACHE_ROOT", "/tmp")
    if any(char.isspace() for char in cache_root):
        raise ValueError("MT_COMPILER_CACHE_ROOT must not contain whitespace (compiler toolchain limitation)")
    cache = Path(tempfile.mkdtemp(prefix="mt-compile-", dir=cache_root))
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache / "inductor")
    os.environ["TRITON_CACHE_DIR"] = str(cache / "triton")
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / f"compiler-cache-rank{rank}.json", {"path": str(cache)})
    if not torch.cuda.is_available() or torch.cuda.device_count() < config["gpus"]:
        raise RuntimeError("Allocated GPUs no longer match configuration")
    instrument(directory)
    options = run["options"].copy()
    loader = {
        "batch_size": config["global_batch_size"] // config["gpus"],
        "num_workers": config["num_workers_per_rank"],
        "data_index": str(root / "data_index.json"),
        "cache": None,
        "splits": ("train", "val"),
        "resample": False,
    }
    if "cuda_prefetch" in options:
        loader["cuda_prefetch"] = options.pop("cuda_prefetch")
    main(
        input=config["parquet"],
        output=str(directory),
        name="model",
        class_spec=str(root / "class_spec.json"),
        builder=HierarchicalBuilder,
        epochs=config["epochs"],
        size=config["size"],
        seed=run["seed"],
        device="cuda",
        dtype="float16",
        ema=False,
        model_builder_kwargs={
            "model_type": "efficientnet_v2_s",
            "weights": str(root / f"initial_seed{run['seed']}.pt"),
            "model_args": {"pretrained": False},
            "hidden": True,
            "normalized": True,
            "droprate": 0.1,
            "fine_tune": False,
        },
        dataloader_builder_kwargs=loader,
        optimizer_builder_kwargs={"optimizer_cls": MuonAuxAdamW, "lr": 0.001, "weight_decay": 0.01},
        criterion_builder_kwargs={"weighted": True, "label_smoothing": None, "weights": [1, 1, 1]},
        regularizer_builder_kwargs={"strength": 0.1},
        lr_schedule_builder_kwargs={"warmup_epochs": 0.25},
        # Predeclare hierarchical columns: otherwise the logger first writes a
        # flat CSV header, then rejects additional columns after validation.
        logger_builder_kwargs={
            "verbose": True,
            "statistics": [
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
            ],
        },
        **options,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "prepare", "train"))
    parser.add_argument("config", type=Path)
    parser.add_argument("--branch", choices=("master", "quant"))
    parser.add_argument("--run")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.action == "preflight":
        preflight(config, args.branch)
    elif args.action == "prepare":
        prepare(config)
    else:
        train(config, args.run)


if __name__ == "__main__":
    main()
