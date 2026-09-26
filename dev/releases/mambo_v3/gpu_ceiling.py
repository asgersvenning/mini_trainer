"""Small Torch GPU-resident throughput reference, reusing a speed-smoke report."""

import argparse
import itertools
import json
import math
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

from deployment.mambo_deploy import Predictor
from deployment.mambo_deploy.preprocessing import prepare_batch
from dev.benchmarks.inference.onnx_inference import file_hash
from dev.releases.mambo_v3.evaluation_data import write_json


def resident_call(predictor, bank, batch_size):
    """Use the deployed CUDA preparation, backbone, head and global native ranks."""
    batches = itertools.cycle(bank.split(batch_size))
    plan = predictor.hierarchy_plan(predictor.selected)

    def call():
        output, _ = predictor._torch(next(batches), False, tensors=True)
        predictor._model_events.clear()  # Diagnostic events must not accumulate between calls.
        return plan.torch_values(output[0].float(), output)[0]

    return call


def measure(call, batch_size, seconds, device):
    # Warm lazy loading, kernels and allocator before calibrating the timed block.
    for _ in range(5):
        values = call()
    torch.cuda.synchronize(device)
    if not all(bool(torch.isfinite(value).all()) for value in values):
        raise RuntimeError("Non-finite resident predictions")
    start = time.perf_counter()
    for _ in range(5):
        call()
    torch.cuda.synchronize(device)
    iterations = max(5, math.ceil(seconds * 5 / (time.perf_counter() - start)))
    torch.cuda.reset_peak_memory_stats(device)
    started = time.time()
    start = time.perf_counter()
    for _ in range(iterations):
        call()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    return {
        "batch_size": batch_size,
        "iterations": iterations,
        "started_unix_seconds": started,
        "finished_unix_seconds": time.time(),
        "elapsed_seconds": elapsed,
        "images_per_second": iterations * batch_size / elapsed,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3,
    }


def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError("This experiment requires CUDA")
    baseline = json.loads(args.baseline.read_text())
    if baseline["settings"]["backend"] != "torch" or baseline["settings"]["tta"] != "none":
        raise ValueError("Use the Torch, no-TTA smoke report")
    args.output.mkdir(parents=True, exist_ok=False)
    settings = baseline["settings"]
    records = baseline["stream_samples"][: max(args.batches)]
    if len(records) < max(args.batches):
        raise ValueError("Baseline report has too few sample images")
    paths = [Path(settings["root"]) / r["path"] for r in records]
    print(f"Checking and preparing {len(paths)} existing sample images once.", flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for record, digest in zip(records, pool.map(file_hash, paths), strict=True):
            if digest != record["sha256"]:
                raise ValueError("Sample image bytes changed")
        host = prepare_batch(paths, pool, compact=True)
    device = "cuda:0"
    predictor = Predictor(settings["bundle"], backend="torch", device=device, precision=settings["precision"], model="full", threads=4)
    if file_hash(predictor.bundle.root / "release.json") != baseline["bundle_sha256"]:
        raise ValueError("Bundle manifest differs from the smoke run")
    bank = torch.from_numpy(host).to(device)
    del host
    report = {
        "status": "running",
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "baseline": str(args.baseline.resolve()),
        "baseline_sha256": file_hash(args.baseline),
        "bundle_sha256": baseline["bundle_sha256"],
        "samples": records,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "precision": predictor.effective_precision,
        "gpu": subprocess.check_output(["nvidia-smi", "-L"], text=True),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device": str(torch.cuda.get_device_properties(device)),
        "boundary": (
            "Resident uint8 input through GPU preprocessing, backbone, head and global hierarchy logits; "
            "no H2D/D2H, CPU prediction objects, TTA or embeddings."
        ),
        "cells": [],
    }
    write_json(args.output / "report.json", report)
    with (args.output / "gpu.csv").open("w") as telemetry:
        monitor = subprocess.Popen(
            [
                "nvidia-smi",
                "--query-gpu=timestamp,uuid,utilization.gpu,utilization.memory,power.draw,memory.used,clocks.sm",
                "--format=csv",
                "-lms",
                "200",
            ],
            stdout=telemetry,
            stderr=subprocess.STDOUT,
        )
        try:
            with torch.inference_mode():
                for size in args.batches:
                    print(f"Batch {size}: warming, then approximately {args.seconds:g}s sustained inference.", flush=True)
                    try:
                        cell = measure(resident_call(predictor, bank, size), size, args.seconds, device)
                    except torch.cuda.OutOfMemoryError:
                        report["cells"].append({"batch_size": size, "status": "out_of_memory"})
                        write_json(args.output / "report.json", report)
                        torch.cuda.empty_cache()
                        print("Memory limit reached; retaining smaller-batch results.", flush=True)
                        break
                    report["cells"].append(cell)
                    write_json(args.output / "report.json", report)
                    print(f"{size}: {cell['images_per_second']:.1f} images/s; {cell['peak_allocated_gib']:.2f} GiB allocated", flush=True)
                completed = [c for c in report["cells"] if "images_per_second" in c]
                if not completed:
                    raise RuntimeError("No batch size fit; try smaller --batches")
                best = max(completed, key=lambda c: c["images_per_second"])
                report["best_batch_size"] = best["batch_size"]
                print(f"Capturing a short trace at batch {best['batch_size']} (excluded from throughput).", flush=True)
                call = resident_call(predictor, bank, best["batch_size"])
                for _ in range(3):
                    call()
                torch.cuda.synchronize(device)
                try:
                    with torch.profiler.profile(
                        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
                    ) as profile:
                        for _ in range(8):
                            call()
                        torch.cuda.synchronize(device)
                    profile.export_chrome_trace(str(args.output / "trace.json.gz"))
                    report["trace"] = "trace.json.gz"
                except RuntimeError as error:
                    report["trace_error"] = str(error)
                    print(f"Profiler unavailable; timing and telemetry retained: {error}", flush=True)
        finally:
            monitor.terminate()
            monitor.wait()
    report["status"] = "complete"
    write_json(args.output / "report.json", report)
    lines = ["batch_size,images_per_second,peak_allocated_gib,peak_reserved_gib"]
    lines.extend(
        f"{c['batch_size']},{c['images_per_second']:.1f},{c['peak_allocated_gib']:.2f},{c['peak_reserved_gib']:.2f}" for c in completed
    )
    (args.output / "summary.csv").write_text("\n".join(lines) + "\n")
    print(f"Done: {args.output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True, help="Existing speed-smoke torch/report.json")
    parser.add_argument("--output", type=Path, required=True, help="New persistent output directory")
    parser.add_argument("--batches", type=int, nargs="+", default=[256, 512, 1024])
    parser.add_argument("--seconds", type=float, default=20, help="Approximate sustained duration per batch size")
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or args.seconds <= 0 or min(args.batches) < 1:
        parser.error("Batch sizes and seconds must be positive and finite")
    args.batches = sorted(set(args.batches))
    if any(max(args.batches) % b for b in args.batches):
        parser.error("Every batch size must divide the largest batch size")
    run(args)


if __name__ == "__main__":
    main()
