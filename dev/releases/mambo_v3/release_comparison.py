"""Run v2 quality or additional v2/v3 timings sequentially in isolated processes."""

import argparse
import json
import os
import subprocess
from pathlib import Path

from dev.releases.mambo_v3.evaluation_data import write_json


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[3]
    shared = ["--manifest", str(args.manifest.resolve()), "--root", str(args.root.resolve())]
    legacy = [
        str(args.v2_python),
        "-P",
        "-m",
        "dev.releases.mambo_v3.legacy_evaluation",
        args.phase,
        "--source",
        str(args.legacy_source.resolve()),
        "--weights",
        str(args.legacy_weights.resolve()),
        *shared,
    ]
    jobs = []
    if args.phase == "full":
        jobs.append(("v2-full", legacy, True))
    else:
        # Reuse existing v3 full/updated-Europe timings; add both northern lists and legacy Europe.
        variants = [(release, device) for device in ("cpu", "cuda:0") for release in ("v2", "v3-torch", "v3-onnx")]
        for trial in range(3):
            for release, device in variants if trial != 1 else reversed(variants):
                name = f"trial-{trial}-{release}-{device.replace(':', '-')}"
                if release == "v2":
                    command = [*legacy, "--device", device]
                    if device == "cpu":
                        command.append("--cpu-float32")
                else:
                    command = [
                        str(args.v3_python),
                        "-m",
                        "dev.releases.mambo_v3.benchmark",
                        *shared,
                        "--bundle",
                        str(args.bundle.resolve()),
                        "--device",
                        device,
                        "--backend",
                        release[3:],
                        "--presets",
                        "north_europe",
                        "north_europe_v3",
                        "europe",
                        "--threads",
                        "4",
                    ]
                    if device == "cpu":
                        command += ["--batches", "1", "8"]
                jobs.append((name, command, release == "v2"))
    plan = {"status": "running", "jobs": jobs, "completed": []}
    write_json(args.output / "plan.json", plan)
    try:
        for name, command, old in jobs:
            env = dict(
                os.environ,
                OMP_NUM_THREADS="4",
                MKL_NUM_THREADS="4",
                OPENBLAS_NUM_THREADS="1",
                CUDA_VISIBLE_DEVICES="0",
                PYTHONHASHSEED="0",
                HF_HUB_OFFLINE="1",
                HF_HUB_CACHE=str(args.hf_cache.resolve()),
                PYTHONPATH=f"{args.legacy_source.resolve()}:{root}" if old else str(root),
            )
            print(name, flush=True)
            with (args.output / f"{name}.log").open("w") as stream:
                subprocess.run(
                    [*command, "--output", str((args.output / name).resolve())],
                    env=env,
                    check=True,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    cwd=root,
                )
            report = json.loads((args.output / name / "report.json").read_text())
            if report["status"] != "complete":
                raise RuntimeError(f"Incomplete comparison: {name}")
            plan["completed"].append(name)
            write_json(args.output / "plan.json", plan)
        plan["status"] = "complete"
    except Exception as error:
        plan.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(args.output / "plan.json", plan)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("full", "benchmark"))
    for name in ("v2-python", "v3-python", "legacy-source", "legacy-weights", "hf-cache", "bundle", "manifest", "root", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    run(parser.parse_args())
