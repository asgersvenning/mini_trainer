"""One small four-variant GPU speed check; no campaign setup or full-data hashing."""

import argparse
import json
import math
import os
import random
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq

from deployment.mambo_deploy import Predictor
from deployment.mambo_deploy.augmentation import DEFAULT_TTA
from dev.benchmarks.inference.onnx_inference import file_hash
from dev.releases.mambo_v3.evaluation_data import write_json

SEED = 20260923


def workers():
    count = len(os.sched_getaffinity(0))
    quota = Path("/sys/fs/cgroup/cpu.max")
    if quota.exists():
        limit, period = quota.read_text().split()
        if limit != "max":
            count = min(count, math.ceil(int(limit) / int(period)))
    return min(count, 48)


def sample(metadata, count=4096):
    table = pq.read_table(metadata, columns=["filename", "set", "speciesKey", "genusKey", "familyKey"])
    table = table.filter(pc.equal(table["set"], "0"))
    if len(table) < count:
        raise ValueError(f"Need {count} test images, found {len(table)}")
    rows = table.take(random.Random(SEED).sample(range(len(table)), count)).to_pylist()
    records = []
    root = metadata.parent.resolve()
    for row in rows:
        path = f"images/{row['speciesKey']}/{row['filename']}"
        if not (root / path).resolve().is_relative_to(root):
            raise ValueError(f"Unsafe image path: {path}")
        records.append({"path": path, "labels": [row[k] for k in ("speciesKey", "genusKey", "familyKey")], "split": "test"})
    return records


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    prepare_workers = args.workers or workers()
    root = args.metadata.resolve().parent
    print("Selecting and warming 4,096 test images; the full dataset is not scanned for images.", flush=True)
    records = sample(args.metadata)
    with ThreadPoolExecutor(max_workers=128) as pool:
        for record, digest in zip(records, pool.map(file_hash, (root / r["path"] for r in records)), strict=True):
            record["sha256"] = digest
    manifest = args.output / "sample.json"
    write_json(manifest, {"schema_version": 1, "dataset": "global-lepi-speed-sample", "records": records})
    write_json(
        args.output / "environment.json",
        {
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "gpu": subprocess.check_output(["nvidia-smi"], text=True),
            "gpu_instances": subprocess.check_output(["nvidia-smi", "-L"], text=True),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cpu_max": Path("/sys/fs/cgroup/cpu.max").read_text() if Path("/sys/fs/cgroup/cpu.max").exists() else None,
            "prepare_workers": prepare_workers,
            "metadata_sha256": file_hash(args.metadata),
            "sample_sha256": file_hash(manifest),
        },
    )
    print("Checking/downloading the standard release weights.", flush=True)
    bundle = Predictor().bundle
    bundle.profile("torch")
    bundle.profile("onnx")
    summary = ["variant,stream_images_per_second,request_images_per_second,peak_host_gib"]
    for backend, tta in (("torch", "none"), ("onnx", "none"), ("torch", DEFAULT_TTA), ("onnx", DEFAULT_TTA)):
        name = backend + ("-tta" if tta != "none" else "")
        print(f"{name}: batch 256, {prepare_workers} preparation workers; three streaming passes", flush=True)
        output = args.output / name
        options = {
            "bundle": bundle.root,
            "manifest": manifest,
            "root": root,
            "output": output,
            "backend": backend,
            "device": "cuda:0",
            "precision": "auto",
            "tta": tta,
            "batches": 256,
            "presets": "full",
            "bank-size": 256,
            "stream-images": 4096,
            "repeats": 3,
            "warmup": 2,
            "threads": 4,
            "stream-workers": prepare_workers,
            "read-workers": 128,
            "read-window": 4096,
            "prefetch-batches": 2,
            "encoded-budget-mib": 1024,
        }
        interpreter = str(args.onnx_python) if backend == "onnx" else sys.executable
        command = [interpreter, "-m", "dev.releases.mambo_v3.benchmark"]
        command.extend(part for key, value in options.items() for part in ("--" + key, str(value)))
        with (args.output / f"{name}.log").open("w") as log:
            try:
                subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError:
                print(f"Failed; inspect {args.output / (name + '.log')}", flush=True)
                raise
        report = json.loads((output / "report.json").read_text())
        cell = report["cells"][0]
        row = (
            f"{name},{cell['streaming']['images_per_second']:.1f},"
            f"{cell['images_per_second']:.1f},{report['peak_rss_kib_linux'] / 1024**2:.2f}"
        )
        summary.append(row)
        (args.output / "summary.csv").write_text("\n".join(summary) + "\n")
        print(row, flush=True)
    print(f"Done: {args.output / 'summary.csv'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--onnx-python", type=Path, required=True, help="Interpreter in the already qualified ONNX environment")
    parser.add_argument("--output", type=Path, required=True, help="New persistent directory under /work")
    parser.add_argument("--workers", type=int, help="Preparation workers; defaults to CPU quota, capped at 48")
    args = parser.parse_args()
    if args.workers is not None and args.workers < 1:
        parser.error("--workers must be positive")
    args.onnx_python = args.onnx_python.absolute()
    if not args.onnx_python.is_file():
        parser.error("--onnx-python must identify an existing interpreter")
    run(args)


if __name__ == "__main__":
    main()
