"""Branch-neutral worker, executed with the selected installed-package interpreter."""

import argparse
import functools
import importlib.metadata
import inspect
import json
import os
import platform
import random
import shutil
import tempfile
import time
from collections import Counter
from pathlib import Path

from compare import digest, plan, uses_quantized_training, write_json


def progress(message):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def qualification_parquet(config):
    """Reservoir-sample each existing split with memory bounded by sample size."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from mini_trainer.integrations.parquet import COLUMNS, iter_parquet_batches, set2split

    excluded = set()
    exclusion = config.get("exclude_qualification")
    if exclusion:
        if digest(exclusion) != config["exclude_qualification_sha256"]:
            raise ValueError("Exclusion sample changed since trial generation")
        excluded = {
            (str(row["speciesKey"]), row["filename"]) for row in pq.read_table(exclusion, columns=["speciesKey", "filename"]).to_pylist()
        }
    settings = config["qualification"]
    rng = random.Random(settings["seed"])
    selected = {split: [] for split in ("train", "validation", "test")}
    seen = Counter()
    scanned = 0
    next_report = 1_000_000
    progress("Selecting qualification rows from existing splits (one streaming Parquet scan)")
    for batch in iter_parquet_batches(config["parquet"]):
        for row in batch.to_pylist():
            if (str(row["speciesKey"]), row["filename"]) in excluded:
                continue
            split = set2split(int(row["set"]))
            seen[split] += 1
            scanned += 1
            reservoir = selected[split]
            limit = settings[split]
            if len(reservoir) < limit:
                reservoir.append((scanned, row))
            else:
                index = rng.randrange(seen[split])
                if index < limit:
                    reservoir[index] = (scanned, row)
        if scanned >= next_report:
            progress(f"Scanned {scanned:,} included rows; retained {sum(map(len, selected.values())):,}")
            next_report = scanned + 1_000_000
    if any(seen[split] < settings[split] for split in selected):
        raise ValueError(f"Not enough rows for requested qualification sample; available: {dict(seen)}")
    rows = [row for _, row in sorted(item for reservoir in selected.values() for item in reservoir)]
    source_schema = pq.read_schema(config["parquet"])
    schema = pa.schema([source_schema.field(column) for column in COLUMNS])
    destination = Path(config["output"]) / "qualification.parquet"
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), destination)
    if any((str(row["speciesKey"]), row["filename"]) in excluded for row in rows):
        raise ValueError("Qualification selection overlaps excluded images")
    write_json(
        Path(config["output"]) / "selection.json",
        {
            "sha256": digest(destination),
            "excluded_sha256": config.get("exclude_qualification_sha256"),
            "excluded_images": len(excluded),
            "eligible_split_counts": dict(seen),
        },
    )
    progress(f"Saved {len(rows):,} qualification rows; original split counts: {dict(seen)}")
    return str(destination)


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
        raise RuntimeError(
            f"Need {config['gpus']} visible CUDA GPUs; available={torch.cuda.is_available()}, "
            f"count={torch.cuda.device_count()}, torch={torch.__version__}, CUDA build={torch.version.cuda}"
        )
    if not verify:
        for index in range(config["gpus"]):
            with torch.cuda.device(index):
                torch.ones(1, device=f"cuda:{index}").add_(1)
                torch.cuda.synchronize()
    if uses_quantized_training(config):
        importlib.metadata.version("torchao")
    if config.get("wandb"):
        importlib.metadata.version("wandb")
    gpu_memory = [torch.cuda.get_device_properties(i).total_memory for i in range(config["gpus"])]
    if config.get("minimum_gpu_memory_gib") and min(gpu_memory) < config["minimum_gpu_memory_gib"] * 2**30:
        raise RuntimeError("GPU memory below qualification requirement; allocate full GPUs, not MIG fractions")
    record = {
        "commit": commit,
        "gpu_memory_bytes": gpu_memory,
        "package": mini_trainer.__file__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "compiler": compiler,
        "cuda": torch.version.cuda,
        "cpu_affinity_count": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "shared_memory_bytes": shutil.disk_usage("/dev/shm").total if Path("/dev/shm").exists() else None,
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
    metadata_path = qualification_parquet(config) if config.get("qualification") else config["parquet"]
    progress("Building taxonomy")
    taxonomy_path = config["parquet"] if config.get("full_taxonomy") else metadata_path
    spec = HierarchicalBuilder.build_class_spec(dir=taxonomy_path, levels=3)
    spec["resize_size"] = config["size"]
    write_json(output / "class_spec.json", spec)
    progress(f"Parsing image metadata; classes per level: {spec['num_classes']}")
    data = get_metadata_from_parquet(metadata_path, cls2idx=spec["cls2idx"])
    if config.get("qualification"):
        # The sampled Parquet is an artifact; images still live beside the source.
        source_root = Path(config["parquet"]).resolve().parent
        sample_root = Path(metadata_path).resolve().parent
        data["path"] = [str(source_root / Path(path).relative_to(sample_root)) for path in data["path"]]
    counts = Counter(data["split"])
    if not all(counts[split] for split in ("train", "validation", "test")):
        raise ValueError(f"Require nonempty existing train/validation/test splits, got {counts}")
    progress(f"Checking duplicates and {len(data['path']):,} image paths")
    if len(set(data["path"])) != len(data["path"]):
        raise ValueError("Duplicate image paths; resolve duplicates/split leakage before comparing")
    checked_at = time.monotonic()
    for index, (path, labels) in enumerate(zip(data["path"], data["class"], strict=True), 1):
        if not Path(path).is_file():
            raise FileNotFoundError(f"Expected Parquet-adjacent images/<speciesKey>/<filename>: {path}")
        if any(value is None for value in labels):
            raise ValueError(f"Unmapped taxonomy: {path}")
        if time.monotonic() - checked_at >= 10:
            progress(f"Checked {index:,}/{len(data['path']):,} image paths")
            checked_at = time.monotonic()
    per_rank = config["global_batch_size"] // config["gpus"]
    if counts["train"] // config["gpus"] < per_rank:
        raise ValueError("Not enough training images for one global batch")
    progress("Saving dataset index")
    write_json(output / "data_index.json", data)
    # Model construction does not need the per-image lists.
    del data
    smoothing = 1 / spec["num_classes"][0]
    progress("Hashing source Parquet and saving dataset provenance")
    write_json(
        output / "dataset.json",
        {
            "parquet_sha256": digest(config["parquet"]),
            "counts": counts,
            "num_classes": spec["num_classes"],
            "taxonomy_scope": "full_source" if config.get("full_taxonomy") else "indexed_rows",
            "label_smoothing_per_level": [1 - (1 - smoothing) ** (1 / (i + 1)) for i in range(3)],
            "image_integrity": "All indexed paths checked; image bytes are not hashed or decoded during preparation. Keep mount immutable.",
            "test_used_for_training_or_selection": False,
            "scope": "qualification_subset" if config.get("qualification") else "full_dataset",
            "qualification": config.get("qualification"),
        },
    )
    for seed in config["seeds"]:
        progress(f"Building CPU FP32 starting model for seed {seed}; math threads={torch.get_num_threads()}")
        progress("Pretrained download and normalized-head initialization may take time; no training is running yet")
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
        progress(f"Saving starting weights for seed {seed}")
        torch.save(model.state_dict(), output / f"initial_seed{seed}.pt")
        (output / "preprocessing.txt").write_text(repr(preprocess) + "\n")
        del model
    progress("Preparation worker finished")


def configure_figures(enabled):
    """Suppress optional evaluation plots in this dedicated worker process."""
    if not enabled:
        from mini_trainer.logging import MultiLogger

        MultiLogger.figures = lambda self, model: None
        progress("Evaluation figures disabled; validation metrics and checkpoints remain enabled")


class TimedLoader:
    """Measure host time waiting for next batches without synchronizing the GPU."""

    def __init__(self, loader, synchronize=None, on_window=None):
        self.loader, self.wait_seconds, self.samples = loader, 0.0, 0
        self.steps, self.windows = 0, []
        self.on_window = on_window or (lambda window: None)
        self.synchronize = synchronize or (lambda: None)

    def __len__(self):
        return len(self.loader)

    def __getattr__(self, key):
        return getattr(self.loader, key)

    def __iter__(self):
        window_start = time.monotonic()
        window_samples, window_steps, window_wait = 0, 0, 0.0
        started = time.monotonic()
        iterator = iter(self.loader)
        while True:
            try:
                batch = next(iterator)
            except StopIteration:
                self.wait_seconds += time.monotonic() - started
                return
            self.wait_seconds += time.monotonic() - started
            self.samples += len(batch[0])
            self.steps += 1
            yield batch
            if self.steps % 32 == 0 or self.steps == len(self.loader):
                self.synchronize()
                now = time.monotonic()
                self.windows.append(
                    {
                        "steps": self.steps - window_steps,
                        "samples": self.samples - window_samples,
                        "seconds": now - window_start,
                        "loader_wait_seconds": self.wait_seconds - window_wait,
                    }
                )
                self.on_window(self.windows[-1])
                window_start, window_samples = now, self.samples
                window_steps, window_wait = self.steps, self.wait_seconds
            started = time.monotonic()


def instrument(output, config=None):
    """Record synchronized phase times per rank without changing optimizer logic."""
    import torch

    import mini_trainer.trainer as trainer
    from mini_trainer.logging import MultiLogger

    # The normal logger resets peak stats within a phase. Disable only that
    # measurement reset so phase-level CUDA peaks include all steps.
    MultiLogger._reset_cuda_memory_stats = staticmethod(lambda: None)
    rank = int(os.environ.get("RANK", "0"))

    def component_timer(function, component):
        @functools.wraps(function)
        def measured(*args, **kwargs):
            started = time.monotonic()
            try:
                return function(*args, **kwargs)
            finally:
                with (output / f"components-rank{rank}.jsonl").open("a") as handle:
                    handle.write(json.dumps({"component": component, "seconds": time.monotonic() - started}) + "\n")

        return measured

    MultiLogger.figures = component_timer(MultiLogger.figures, "figures")
    trainer.save_on_master = component_timer(trainer.save_on_master, "checkpoint")

    def wrap(function, phase):
        @functools.wraps(function)
        def measured(*args, **kwargs):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            bound = inspect.signature(function).bind(*args, **kwargs)

            def save_window(window):
                with (output / f"windows-rank{rank}.jsonl").open("a") as handle:
                    handle.write(json.dumps({"phase": phase, "epoch": bound.arguments["epoch"], **window}) + "\n")

            loader = TimedLoader(bound.arguments["data_loader"], torch.cuda.synchronize, save_window)
            bound.arguments["data_loader"] = loader
            value = function(*bound.args, **bound.kwargs)
            torch.cuda.synchronize()
            record = {
                "phase": phase,
                "epoch": bound.arguments["epoch"],
                "samples": loader.samples,
                "steps": loader.steps,
                "windows": loader.windows,
                "loader_wait_seconds": loader.wait_seconds,
                "seconds": time.monotonic() - started,
                "max_memory_allocated": torch.cuda.max_memory_allocated(),
                "max_memory_reserved": torch.cuda.max_memory_reserved(),
            }
            if phase == "validation":
                record["selection_metric"] = float(value)
            with (output / f"phases-rank{rank}.jsonl").open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            if config and config.get("wandb"):
                import torch.distributed as dist
                import wandb

                totals = torch.tensor(
                    [record["seconds"], record["max_memory_allocated"], record["max_memory_reserved"], loader.wait_seconds],
                    device="cuda",
                    dtype=torch.float64,
                )
                if dist.is_initialized():
                    dist.all_reduce(totals, op=dist.ReduceOp.MAX)
                if rank == 0:
                    seconds, allocated, reserved, wait = totals.tolist()
                    wandb.log(
                        {
                            f"qualification/{phase}_images_per_second": loader.samples * config["gpus"] / seconds,
                            f"qualification/{phase}_seconds": seconds,
                            f"qualification/{phase}_peak_allocated_bytes": allocated,
                            f"qualification/{phase}_peak_reserved_bytes": reserved,
                            f"qualification/{phase}_loader_wait_seconds": wait,
                            "qualification/epoch": bound.arguments["epoch"],
                        }
                    )
                    write_json(output / "wandb-run.json", {"id": wandb.run.id, "url": wandb.run.url})
            return value

        return measured

    trainer.train_one_epoch = wrap(trainer.train_one_epoch, "train")
    trainer.evaluate = wrap(trainer.evaluate, "validation")


def configure_wandb(config, name):
    if not config.get("wandb"):
        return {}
    # Authentication remains the existing W&B integration's responsibility.
    # A trial-specific shared ID keeps eight ranks together and trials separate.
    import hashlib

    from mini_trainer.logging import MetricLogger, WandbLogger

    run_id = hashlib.sha256(f"{config['output']}/{name}".encode()).hexdigest()[:16]
    return {
        "logger_cls": [MetricLogger, WandbLogger],
        "logger_cls_extra_kwargs": [
            {},
            {
                "project": os.environ.get("WANDB_PROJECT", "mini-trainer-ddp"),
                "run_name": f"{Path(config['output']).name}-{name}",
                "run_id": run_id,
            },
        ],
    }


def configure_training_checks(config, directory):
    """Check restored state before DDP wrapping/compilation and optimizer updates."""
    import torch

    import mini_trainer.train as train_module

    original = train_module.train

    def equal(actual, expected):
        if isinstance(expected, torch.Tensor):
            assert actual.dtype == expected.dtype and torch.equal(actual.detach().cpu(), expected)
        elif isinstance(expected, dict):
            assert actual.keys() == expected.keys()
            for key in expected:
                equal(actual[key], expected[key])
        elif isinstance(expected, (list, tuple)):
            assert len(actual) == len(expected)
            for a, b in zip(actual, expected, strict=True):
                equal(a, b)
        else:
            assert actual == expected

    def checked(**kwargs):
        kwargs["weight_store_rate"] = 1
        if config.get("checkpoint"):
            saved = torch.load(config["checkpoint"], map_location="cpu", weights_only=False)
            assert kwargs["start_epoch"] == saved["epoch"] + 1 == config["resume_epoch"]
            for key in ("model", "optimizer", "lr_scheduler", "scaler"):
                if key == "scaler" and kwargs[key] is None:
                    continue
                equal(kwargs[key].state_dict(), saved[key])
            write_json(
                directory / f"restore-rank{os.environ.get('RANK', '0')}.json",
                {
                    "checkpoint_sha256": digest(config["checkpoint"]),
                    "start_epoch": kwargs["start_epoch"],
                    "model_optimizer_scheduler_scaler_restored": True,
                },
            )
            del saved
        return original(**kwargs)

    train_module.train = checked


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
    configure_figures(config.get("figures", True))
    instrument(directory, config)
    options = run["options"].copy()
    logging_options = configure_wandb(config, name)
    if config.get("checkpoint") or config.get("mode") == "scaling":
        configure_training_checks(config, directory)
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
        checkpoint=config.get("checkpoint"),
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
            **logging_options,
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
    if config.get("wandb"):
        import wandb

        wandb.finish()


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
