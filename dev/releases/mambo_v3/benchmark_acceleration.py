"""Fresh-process CPU/GPU timings for the qualified automatic deployment settings."""

import argparse
import json
import os
import subprocess
from pathlib import Path

from deployment.mambo_deploy.augmentation import DEFAULT_TTA, PROFILES
from dev.releases.mambo_v3.evaluation_data import write_json


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "completed": [], "commands": []}
    shared = ["--bundle", str(args.bundle.resolve()), "--manifest", str(args.manifest.resolve()), "--root", str(args.root.resolve())]
    shared += ["--tta", args.tta]
    variants = [(backend, device) for device in ("cuda:0", "cpu") for backend in ("torch", "onnx")]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1", PYTHONHASHSEED="0")
    try:
        for trial in range(3):
            for backend, device in variants if trial != 1 else reversed(variants):
                name = f"trial-{trial}-{backend}-{device.replace(':', '-')}"
                command = [
                    str(args.python),
                    "-m",
                    "dev.releases.mambo_v3.benchmark",
                    *shared,
                    "--backend",
                    backend,
                    "--device",
                    device,
                    "--precision",
                    "auto",
                    "--threads",
                    "4",
                    "--presets",
                    *args.presets,
                    "--batches",
                    "1",
                    "8",
                ]
                if device != "cpu":
                    command += ["32"]
                command += ["--output", str((args.output / name).resolve())]
                report["commands"].append(command)
                write_json(args.output / "plan.json", report)
                print(name, flush=True)
                with (args.output / f"{name}.log").open("w") as stream:
                    subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
                if json.loads((args.output / name / "report.json").read_text())["status"] != "complete":
                    raise ValueError("Incomplete trial")
                report["completed"].append(name)
                write_json(args.output / "plan.json", report)
        report["status"] = "complete"
    except Exception as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        write_json(args.output / "plan.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("python", "bundle", "manifest", "root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--tta", nargs="?", const=DEFAULT_TTA, choices=PROFILES, default="none")
    parser.add_argument("--presets", nargs="+", default=["north_europe", "europe", "full"])
    run(parser.parse_args())
