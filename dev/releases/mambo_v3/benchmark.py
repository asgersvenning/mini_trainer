"""Fresh-process local deployment timings; cold first-call and warmed boundaries are explicit."""

import argparse
import os
import resource
import statistics
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np

from deployment.mambo_deploy import Predictor
from deployment.mambo_deploy.preprocessing import preprocess
from dev.benchmarks.inference.onnx_inference import file_hash
from dev.releases.mambo_v3.evaluate import runtime_settings
from dev.releases.mambo_v3.evaluation_data import load_records, write_json


def snapshot():
    cpuinfo = Path("/proc/cpuinfo")
    result = {
        "cpu_model": next(
            (line.split(":", 1)[1].strip() for line in cpuinfo.read_text().splitlines() if line.startswith("model name")), None
        )
        if cpuinfo.exists()
        else None
    }
    for name, command in (
        (
            "gpu",
            ["nvidia-smi", "--query-gpu=name,memory.used,power.draw,temperature.gpu,clocks.current.sm,pstate", "--format=csv,noheader"],
        ),
        ("gpu_processes", ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"]),
    ):
        try:
            result[name] = subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL, timeout=10).strip()
        except (OSError, subprocess.SubprocessError):
            result[name] = None
    for name, path in (
        ("ac_online", "/sys/class/power_supply/AC1/online"),
        ("power_profile", "/sys/firmware/acpi/platform_profile"),
        ("cpu_governor", "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
    ):
        result[name] = Path(path).read_text().strip() if Path(path).is_file() else None
    return result


def timing(call, repeats):
    values = []
    for _ in range(repeats):
        t = time.perf_counter()
        call()
        values.append(time.perf_counter() - t)
    return {"seconds": values, "median_seconds": statistics.median(values), "p95_seconds": float(np.percentile(values, 95))}


@contextmanager
def observe_loading(backend, values):
    def timed(name, function):
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                values[name] = values.get(name, 0.0) + time.perf_counter() - start

        return wrapper

    if backend == "torch":
        import torch

        from mini_trainer.builders import BaseBuilder
        from mini_trainer.modeling.classifier import Classifier

        with (
            patch.object(torch, "load", timed("checkpoint_deserialization", torch.load)),
            patch.object(BaseBuilder, "build_model", timed("architecture_and_weight_construction", BaseBuilder.build_model)),
            patch.object(
                Classifier,
                "init_spherical_repulsion",
                classmethod(timed("spherical_initialization_within_model_build", Classifier.init_spherical_repulsion.__func__)),
            ),
        ):
            yield
    else:
        import onnxruntime as ort

        with patch.object(ort, "InferenceSession", timed("session_construction", ort.InferenceSession)):
            yield


def benchmark(args):
    args.output.mkdir(parents=True, exist_ok=False)
    setup_start = time.perf_counter()
    settings = runtime_settings(args.threads, args.backend)
    setup_seconds = time.perf_counter() - setup_start
    report = {
        "status": "running",
        "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "runtime": settings,
        "runtime_import_config_seconds": setup_seconds,
        "before": snapshot(),
        "pid": os.getpid(),
        "runner_sha256": file_hash(__file__),
        "bundle_sha256": file_hash(args.bundle / "release.json"),
        "manifest_sha256": file_hash(args.manifest),
        "cells": [],
        "boundaries": {
            "end_to_end": "image path through CPU result/embedding; decode/preprocess/transfer/reduction included",
            "prepared": "preprocessed CPU tensor through CPU leaf scores/embeddings; transfers included; no decode or reduction",
            "cold": "first image after Predictor construction; lazy model/session load included; runtime import/config measured separately",
        },
    }
    try:
        _, records = load_records(args.manifest, args.root, max(32, max(args.batches)), args.seed)
        report["samples"] = records
        t = time.perf_counter()
        predictor = Predictor(
            args.bundle, backend=args.backend, device=args.device, model="full", threads=args.threads, batch_size=max(args.batches)
        )
        report["constructor_seconds"] = time.perf_counter() - t
        paths = [args.root / r["path"] for r in records]
        for path, record in zip(paths, records, strict=True):
            if file_hash(path) != record["sha256"]:
                raise ValueError("Benchmark image bytes changed")
        predict = predictor.predict_with_embeddings if args.embeddings else predictor.predict
        report["load_components_seconds"] = {}
        cold_start = time.perf_counter()
        with observe_loading(args.backend, report["load_components_seconds"]):
            predict(paths[:1])
        report["cold_first_image_seconds"] = time.perf_counter() - cold_start
        runtime = predictor._torch if args.backend == "torch" else predictor._onnx
        for size in args.batches:
            prepared = np.stack([preprocess(path) for path in paths[:size]])
            for preset in ("full", "europe_v3"):
                selector = Predictor(args.bundle, model=preset)
                predictor._apply_class_mask(selector.class_list)
                for _ in range(args.warmup):
                    predict(paths[:size])
                    runtime(prepared, args.embeddings)
                cell = {"batch_size": size, "preset": preset, "list_sha256": selector.class_list_sha256}
                cell["preprocessing"] = timing(lambda: np.stack([preprocess(path) for path in paths[:size]]), args.repeats)
                cell["end_to_end"] = timing(lambda: predict(paths[:size]), args.repeats)
                cell["prepared"] = timing(lambda: runtime(prepared, args.embeddings), args.repeats)
                cell["images_per_second"] = size / cell["end_to_end"]["median_seconds"]
                cell["resources"] = snapshot()
                report["cells"].append(cell)
                print(args.backend, args.device, args.embeddings, size, preset, round(cell["images_per_second"], 2), flush=True)
        if args.backend == "onnx":
            import onnxruntime as ort

            report["runtime"]["onnxruntime"] = ort.__version__
            report["providers"] = {name: session.get_providers() for name, session in predictor._sessions.items()}
            report["provider_options"] = {key: session.get_provider_options() for key, session in predictor._sessions.items()}
        if args.device != "cpu" and args.backend == "torch":
            import torch

            report["torch_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            report["torch_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        report["status"] = "complete"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["torch_imported"] = "torch" in sys.modules
        report["peak_rss_kib_linux"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report["after"] = snapshot()
        write_json(args.output / "report.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "manifest", "root", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--backend", choices=["torch", "onnx"], required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--embeddings", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 8, 32])
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20260923)
    args = parser.parse_args()
    if min(args.batches) < 1 or args.warmup < 1 or args.repeats < 3:
        parser.error("Positive batches/warmup and at least three repeats required")
    benchmark(args)


if __name__ == "__main__":
    main()
