"""Compose TensorRT held-out quality, inspection, paired latency and isolated memory."""

import json
import os
import subprocess
import sys
from argparse import ArgumentParser
from collections import Counter
from pathlib import Path

from .inference_pair import run_pair
from .onnx_inference import file_hash


def build_bundle(path):
    """Bind retained inspection to the exact engine measured by every stage."""
    path = Path(path).resolve(strict=True)
    report = json.loads((path / "report.json").read_text())
    engine, layers = path / "model.engine", path / "layers.json"
    if report["status"] != "passed" or file_hash(engine) != report["engine"]["sha256"]:
        raise ValueError("Build report must describe the exact successful engine")
    if file_hash(layers) != report["engine"]["layers_sha256"]:
        raise ValueError("Layer inspection differs from the build report")
    inspection = json.loads(layers.read_text())["Layers"]
    return engine, {
        "directory": str(path),
        "report_sha256": file_hash(path / "report.json"),
        "engine": report["engine"],
        "settings": report["settings"],
        "weight_types": dict(Counter(layer["Weights"]["Type"] for layer in inspection if "Weights" in layer)),
        "gemms": [layer for layer in inspection if layer.get("LayerType") == "gemm"],
    }


def evaluate(
    baseline_build,
    candidate_build,
    manifest,
    inputs,
    output,
    trials=3,
    warmup=10,
    repeats=31,
    memory_runs=20,
    threads=1,
    device=0,
    pinned=False,
    metrics_python=sys.executable,
):
    if min(trials, repeats, memory_runs, threads) < 1 or min(warmup, device) < 0:
        raise ValueError("Require positive trials/repeats/memory_runs/threads and nonnegative warmup/device")
    manifest, inputs = Path(manifest).resolve(strict=True), Path(inputs).resolve(strict=True)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": 1,
        "status": "running",
        "runner_sha256": file_hash(__file__),
        "manifest": {"path": str(manifest), "sha256": file_hash(manifest)},
        "inputs": {"path": str(inputs), "sha256": file_hash(inputs)},
        "settings": {
            "trials": trials,
            "warmup": warmup,
            "repeats": repeats,
            "memory_runs": memory_runs,
            "threads": threads,
            "device": device,
            "pinned_host_io": pinned,
        },
        "builds": {},
        "stages": [],
        "trials": [],
        "scope": (
            "Paired held-out mini_metrics quality, build-bound inspection, adjacent paired host latency and isolated memory. "
            "Evaluated is not production acceptance; inspect all five metrics, undefined values and resource trade-offs. "
            "Latency excludes preprocessing/loading; device memory is device-wide snapshots, not per-process or transient peaks. "
            "Run on quiescent target hardware; no automatic thermal control or disk-cache eviction."
        ),
    }

    def save():
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    def child(name, module, arguments):
        command = [sys.executable, "-m", f"dev.benchmarks.inference.{module}", *map(str, arguments), "--output", str(output / name)]
        stage = {"name": name, "command": command, "status": "running", "log": f"{name}.log"}
        report["stages"].append(stage)
        save()
        with (output / stage["log"]).open("w") as log:
            process = subprocess.run(
                command,
                cwd=Path(__file__).resolve().parents[3],
                env={**os.environ, "PYTHONHASHSEED": "0", "OMP_NUM_THREADS": str(threads)},
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        stage["returncode"] = process.returncode
        if process.returncode:
            raise RuntimeError(f"{name} failed; see {output / stage['log']}")
        result = json.loads((output / name / "report.json").read_text())
        if result["status"] != "passed":
            raise RuntimeError(f"Unexpected {name} status: {result['status']}")
        stage.update(status="passed", report_sha256=file_hash(output / name / "report.json"))
        return result

    def identity(role, engine_info):
        if engine_info["sha256"] != report["builds"][role]["engine"]["sha256"]:
            raise ValueError(f"{role} engine changed between build and evaluation stages")

    def resource_inputs(result):
        if result["inputs"]["sha256"] != report["inputs"]["sha256"]:
            raise ValueError("Resource inputs changed between stages")

    try:
        report["phase"] = "inspection"
        engines = {}
        for role, path in (("baseline", baseline_build), ("candidate", candidate_build)):
            engines[role], report["builds"][role] = build_bundle(path)
        report["phase"] = "quality"
        save()
        runtime = {"backend": "tensorrt", "device": device}
        quality = run_pair(manifest, engines["baseline"], engines["candidate"], output / "quality", runtime, runtime, metrics_python)
        report["quality"] = {
            "report_sha256": file_hash(output / "quality/report.json"),
            "levels": quality["levels"],
            "undefined_metrics": quality["undefined_metrics"],
            "models": quality["models"],
        }
        collectors = {}
        for role in engines:
            result = json.loads((output / f"quality/{role}/report.json").read_text())
            if len(result["model_files"]) != 1:
                raise ValueError("Require one TensorRT engine per collector")
            identity(role, result["model_files"][0])
            if result["manifest"]["sha256"] != report["manifest"]["sha256"]:
                raise ValueError("Quality manifest changed between stages")
            collectors[role] = result
        if collectors["baseline"]["batches"] != collectors["candidate"]["batches"]:
            raise ValueError("Quality input batches differ between engines")
        if collectors["baseline"]["runtime"] != collectors["candidate"]["runtime"]:
            raise ValueError("Quality runtime environments differ")
        report["phase"] = "resources"
        save()
        flags = ["--device", device, *(["--pinned"] if pinned else [])]
        for trial in range(trials):
            record = {"trial": trial}
            report["trials"].append(record)
            latency = child(
                f"latency-{trial}",
                "tensorrt_pair",
                [
                    "--baseline",
                    engines["baseline"],
                    "--candidate",
                    engines["candidate"],
                    "--inputs",
                    inputs,
                    "--warmup",
                    warmup,
                    "--repeats",
                    repeats,
                    *flags,
                    *(["--reverse"] if trial % 2 else []),
                ],
            )
            resource_inputs(latency)
            for role in engines:
                identity(role, latency["models"][role])
            record["latency"] = latency["summary"]
            memory = {}
            order = ["baseline", "candidate"] if trial % 2 == 0 else ["candidate", "baseline"]
            record["memory_order"] = order
            for role in order:
                result = child(
                    f"memory-{trial}-{role}",
                    "tensorrt_memory",
                    ["--engine", engines[role], "--inputs", inputs, "--runs", memory_runs, "--threads", threads, *flags],
                )
                resource_inputs(result)
                identity(role, result["engine"])
                memory[role] = result
                record.setdefault("memory", {})[role] = result["memory"]
            a, b = memory["baseline"], memory["candidate"]
            if any(a[key] != b[key] for key in ("settings", "versions", "environment")):
                raise ValueError("Memory trial environments or settings differ")
            for value in memory.values():
                if any(value["versions"][key] != latency["versions"][key] for key in ("torch", "tensorrt", "numpy")):
                    raise ValueError("Latency and memory runtime versions differ")
                if value["environment"]["gpu"] != latency["environment"]["gpu"]:
                    raise ValueError("Latency and memory GPU identities differ")
            if any(latency["versions"][key] != collectors["baseline"]["runtime"][key] for key in ("tensorrt", "torch")):
                raise ValueError("Quality and resource runtime versions differ")
            if latency["environment"]["gpu"] != collectors["baseline"]["runtime"]["gpu"]:
                raise ValueError("Quality and resource GPU identities differ")
            save()
        lines = [
            "# TensorRT deployment comparison",
            "",
            report["scope"],
            "",
            "| Trial | Paired latency ratio | Baseline warm device bytes | Candidate warm device bytes | "
            "Baseline host RSS bytes | Candidate host RSS bytes |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for trial in report["trials"]:
            a, b = [trial["memory"][role]["warm"] for role in ("baseline", "candidate")]
            lines.append(
                f"| {trial['trial']} | {trial['latency']['median_paired_ratio']:.4f} | "
                f"{a['device_used_bytes']} | {b['device_used_bytes']} | "
                f"{a['host']['resident_bytes']} | {b['host']['resident_bytes']} |"
            )
        lines.extend(
            [
                "",
                "Latency ratios are candidate / baseline. Device readings include shared/runtime overhead; "
                "inspect initialization snapshots in report.json.",
                "",
                (output / "quality/summary.md").read_text(),
            ]
        )
        (output / "summary.md").write_text("\n".join(lines) + "\n")
        report.update(status="evaluated", phase="complete")
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        for stage in report["stages"]:
            if stage["status"] == "running":
                stage["status"] = "failed"
        raise
    finally:
        save()
    return report


def main():
    parser = ArgumentParser(description=__doc__)
    for name in ("baseline-build", "candidate-build", "manifest", "inputs", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name, default in (("trials", 3), ("warmup", 10), ("repeats", 31), ("memory-runs", 20), ("threads", 1), ("device", 0)):
        parser.add_argument("--" + name, type=int, default=default)
    parser.add_argument("--pinned", action="store_true")
    parser.add_argument("--metrics-python", default=sys.executable)
    evaluate(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
