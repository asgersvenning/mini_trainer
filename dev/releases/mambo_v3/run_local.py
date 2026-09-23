"""Run a bounded release phase sequentially; each variant/trial uses a fresh process."""

import argparse
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

from dev.releases.mambo_v3.evaluation_data import write_json


def plan(args):
    shared = ["--bundle", str(args.bundle.resolve()), "--manifest", str(args.manifest.resolve()), "--root", str(args.root.resolve())]
    jobs = []
    if args.phase in ("qualification", "full"):
        variants = (
            itertools.product(("cpu", "cuda:0"), ("torch", "onnx"), (False, True))
            if args.phase == "qualification"
            else [("cuda:0", "torch", False), ("cuda:0", "onnx", False)]
        )
        for device, backend, embeddings in variants:
            name = f"{backend}-{device.replace(':', '-')}-{'embedding' if embeddings else 'prediction'}"
            command = [
                str(args.python),
                "-m",
                "dev.releases.mambo_v3.evaluate",
                "collect",
                *shared,
                "--backend",
                backend,
                "--device",
                device,
                "--batch-size",
                "32",
                "--threads",
                "4",
            ]
            if embeddings:
                command.append("--embeddings")
            if args.phase == "qualification":
                command += ["--count", str(args.count)]
            jobs.append((name, command))
    else:
        variants = list(itertools.product(("cpu", "cuda:0"), ("torch", "onnx"), (False, True)))
        for trial in range(3):
            for device, backend, embeddings in variants if trial % 2 == 0 else list(reversed(variants)):
                for threads in [4, 1] if device == "cpu" else [4]:
                    name = f"trial-{trial}-{backend}-{device.replace(':', '-')}-{'embedding' if embeddings else 'prediction'}-t{threads}"
                    command = [
                        str(args.python),
                        "-m",
                        "dev.releases.mambo_v3.benchmark",
                        *shared,
                        "--backend",
                        backend,
                        "--device",
                        device,
                        "--threads",
                        str(threads),
                    ]
                    if embeddings:
                        command.append("--embeddings")
                    # Single-thread runs qualify interactive latency; practical thread runs sweep all batches.
                    if threads == 1:
                        command += ["--batches", "1"]
                    elif device == "cpu":
                        command += ["--batches", "1", "8"]
                    jobs.append((name, command))
    return [(name, [*command, "--output", str((args.output / name).resolve())]) for name, command in jobs]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["qualification", "full", "benchmark"])
    for name in ("bundle", "manifest", "root", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--count", type=int, default=256)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    jobs = plan(args)
    record = {"phase": args.phase, "status": "running", "commands": jobs, "completed": []}
    write_json(args.output / "plan.json", record)
    environment = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="0",
        PYTHONHASHSEED="0",
        OMP_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="1",
        MPLBACKEND="Agg",
    )
    try:
        for name, command in jobs:
            print(name, flush=True)
            with (args.output / f"{name}.log").open("w") as stream:
                subprocess.run(command, env=environment, stdout=stream, stderr=subprocess.STDOUT, check=True)
            report = json.loads((args.output / name / "report.json").read_text())
            if report["status"] != "complete":
                raise RuntimeError(f"Incomplete variant: {name}")
            record["completed"].append(name)
            write_json(args.output / "plan.json", record)
        record["status"] = "complete"
    except Exception as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(args.output / "plan.json", record)


if __name__ == "__main__":
    main()
