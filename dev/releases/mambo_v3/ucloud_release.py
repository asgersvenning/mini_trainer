"""Plan and run the five release pipelines on an allocated UCloud node."""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

from dev.benchmarks.inference.onnx_inference import file_hash
from dev.releases.mambo_v3.evaluation_data import write_json

ROOT = Path(__file__).resolve().parents[3]
RECIPE = "rotation30_pad25_3"
VARIANTS = ("v2", "torch", "onnx", "torch-tta", "onnx-tta")
PATHS = ("v2_python", "v3_python", "metrics_python", "legacy_source", "legacy_weights", "hf_cache", "bundle", "manifest", "root", "output")


def configuration(path):
    data = json.loads(path.read_text())
    for key in (*PATHS, "onnx_python", "timing_manifest", "timing_root"):
        if key in data:
            value = path.parent / Path(data[key]).expanduser()
            # Resolving a venv Python symlink selects the base interpreter and loses its packages.
            data[key] = os.path.abspath(value) if key.endswith("_python") else str(value.resolve())
    for key in ("quality_batch_size", "qualification_count", "threads"):
        if not isinstance(data[key], int) or data[key] < 1:
            raise ValueError(f"Positive integer required: {key}")
    for key in ("cpu_batches", "gpu_batches"):
        if not data[key] or any(not isinstance(b, int) or b < 1 for b in data[key]):
            raise ValueError(f"Positive batches required: {key}")
    for key in ("quality_presets", "timing_presets"):
        if not data[key] or not set(data[key]) <= {"full", "europe", "north_europe"}:
            raise ValueError("Use presets shared with V2")
    if "full" not in data["quality_presets"]:
        raise ValueError("In-domain quality must include the global vocabulary")
    if not data["environment_id"] or not data["timing_devices"] or not set(data["timing_devices"]) <= {"cpu", "cuda:0"}:
        raise ValueError("Require environment identity and CPU/CUDA timing devices")
    if data["quality_device"] not in ("cpu", "cuda:0"):
        raise ValueError("Unsupported quality device")
    return data


def new_campaign(config, output):
    """Reuse prepared assets with the active interpreter and a fresh results directory."""
    updated = dict(config)
    for key in ("v2_python", "v3_python", "metrics_python"):
        updated[key] = os.path.abspath(sys.executable)
    output = output.expanduser().resolve()
    updated["output"] = str(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", updated)
    return updated


def jobs(config, phase):
    """Pure plan construction: no dataset access, imports of runtimes, or execution."""
    timing = phase == "benchmark"
    manifest = config.get("timing_manifest", config["manifest"]) if timing else config["manifest"]
    root = config.get("timing_root", config["root"]) if timing else config["root"]
    shared = ["--manifest", manifest, "--root", root, "--threads", str(config["threads"])]
    bank_size = max(32, *config["cpu_batches"], *config["gpu_batches"])
    planned = []
    for trial in range(3 if timing else 1):
        variants = list(VARIANTS)
        if trial == 1:
            variants.reverse()
        devices = config["timing_devices"] if timing else [config["quality_device"]]
        for device in devices:
            for variant in variants:
                legacy = variant == "v2"
                if legacy:
                    command = [
                        config["v2_python"],
                        "-P",
                        "-m",
                        "dev.releases.mambo_v3.legacy_evaluation",
                        phase,
                        "--source",
                        config["legacy_source"],
                        "--weights",
                        config["legacy_weights"],
                    ]
                    if device == "cpu":
                        command += ["--cpu-float32"]
                else:
                    module = "benchmark" if timing else "evaluate"
                    interpreter = config.get("onnx_python", config["v3_python"]) if variant.startswith("onnx") else config["v3_python"]
                    command = [interpreter, "-m", f"dev.releases.mambo_v3.{module}"]
                    if not timing:
                        command += ["collect", "--decode-workers", str(config["threads"])]
                    command += [
                        "--bundle",
                        config["bundle"],
                        "--backend",
                        variant.split("-")[0],
                        "--precision",
                        "auto",
                        "--tta",
                        RECIPE if variant.endswith("-tta") else "none",
                    ]
                command += [*shared, "--device", device, "--presets", *config["timing_presets" if timing else "quality_presets"]]
                if timing:
                    command += [
                        "--batches",
                        *map(str, config["cpu_batches"] if device == "cpu" else config["gpu_batches"]),
                        "--bank-size",
                        str(bank_size),
                    ]
                else:
                    command += ["--batch-size", str(config["quality_batch_size"])]
                    if phase == "qualification":
                        command += ["--count", str(config["qualification_count"])]
                name = f"trial-{trial}-{variant}-{device.replace(':', '-')}" if timing else variant
                command += ["--output", str(Path(config["output"]) / phase / name)]
                planned.append({"name": name, "variant": variant, "device": device, "legacy": legacy, "command": command})
    return planned


def runtime_environments(config):
    """Record installed environments, independently of how they were resolved."""
    script = (
        "import importlib.metadata as m, json, sys; "
        "print(json.dumps({'python': sys.version, 'packages': sorted("
        "[d.metadata['Name'], d.version, d.read_text('direct_url.json')] "
        "for d in m.distributions())}))"
    )
    return {
        interpreter: json.loads(subprocess.check_output([interpreter, "-I", "-c", script], text=True))
        for interpreter in sorted({config[key] for key in ("v2_python", "v3_python", "metrics_python", "onnx_python") if key in config})
    }


def fingerprint(config):
    manifest = json.loads(Path(config["manifest"]).read_text())
    if manifest.get("dataset") != "global-lepi-test" or len(manifest["records"]) != 632913:
        raise ValueError("Require the verified original 632,913-image UCloud test manifest")
    if any(r["split"] != "test" for r in manifest["records"]) or manifest.get("provenance", {}).get("test_set") != "0":
        raise ValueError("Preserve original test membership; no resplitting")
    hashes = {key: file_hash(config[key]) for key in ("manifest", "timing_manifest") if key in config}
    hashes["bundle"] = file_hash(Path(config["bundle"]) / "release.json")
    hashes["scripts"] = {
        str(p.relative_to(ROOT)): file_hash(p)
        for folder in (ROOT / "dev/releases/mambo_v3", ROOT / "deployment/mambo_deploy")
        for p in sorted(folder.glob("*.py"))
    }
    hashes["environments"] = runtime_environments(config)
    hashes["revision"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    return {"config": config, "inputs": hashes}


def validated_report(directory):
    report = json.loads((directory / "report.json").read_text())
    if report["status"] != "complete":
        raise ValueError(f"Incomplete job: {directory}; preserve it elsewhere before resuming")
    for preset, digest in report.get("csv_sha256", {}).items():
        if file_hash(directory / preset / "mini_metric.csv") != digest:
            raise ValueError(f"Changed predictions: {directory}")
    return report


def bank_identity(directory, report):
    records = report["samples"]
    if isinstance(records, int):
        records = json.loads((directory / "samples.json").read_text())
        if len(records) != report["samples"] or file_hash(directory / "samples.json") != report["sample_ids_sha256"]:
            raise ValueError("Changed legacy timing image bank")
    return hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()


def run(config, phase, resume=False):
    frozen = fingerprint(config)
    output = Path(config["output"]) / phase
    planned = jobs(config, phase)
    if phase in ("full", "benchmark"):
        prior = json.loads((Path(config["output"]) / "qualification/plan.json").read_text())
        if prior["status"] != "complete" or prior["fingerprint"] != frozen:
            raise ValueError("Complete qualification with this exact configuration and revision first")
        for job in prior["jobs"]:
            directory = Path(config["output"]) / "qualification" / job["name"]
            validated_report(directory)
            if file_hash(directory / "report.json") != prior["reports_sha256"][job["name"]]:
                raise ValueError("Qualification report changed")
    if output.exists():
        if not resume:
            raise FileExistsError("Use a fresh output or --resume to verify and reuse completed jobs")
        plan = json.loads((output / "plan.json").read_text())
        if plan["fingerprint"] != frozen or plan["jobs"] != planned:
            raise ValueError("Changed campaign configuration, inputs or code; use a fresh output")
    else:
        output.mkdir(parents=True)
        plan = {
            "fingerprint": frozen,
            "environment_id": config["environment_id"],
            "platform": platform.platform(),
            "jobs": planned,
            "completed": [],
            "reports_sha256": {},
        }
    plan.pop("error", None)
    plan["status"] = "running"
    write_json(output / "plan.json", plan)
    identities = set()
    try:
        for job in planned:
            directory = output / job["name"]
            if directory.exists() and not (directory / "report.json").exists():
                raise ValueError(f"Partial job {directory}; preserve it elsewhere before resuming")
            if not directory.exists():
                env = dict(
                    os.environ,
                    CUDA_VISIBLE_DEVICES=config["cuda_visible_devices"],
                    OMP_NUM_THREADS=str(config["threads"]),
                    MKL_NUM_THREADS=str(config["threads"]),
                    OPENBLAS_NUM_THREADS="1",
                    PYTHONHASHSEED="0",
                    HF_HUB_OFFLINE="1",
                    HF_HUB_CACHE=config["hf_cache"],
                    PYTHONPATH=f"{config['legacy_source']}:{ROOT}" if job["legacy"] else str(ROOT),
                )
                print(job["name"], flush=True)
                with (output / f"{job['name']}.log").open("w") as stream:
                    try:
                        subprocess.run(job["command"], cwd=ROOT, env=env, check=True, stdout=stream, stderr=subprocess.STDOUT)
                    except subprocess.CalledProcessError:
                        log = output / f"{job['name']}.log"
                        print(f"Job failed; log: {log}\n" + "\n".join(log.read_text(errors="replace").splitlines()[-25:]), flush=True)
                        raise
            report = validated_report(directory)
            digest = file_hash(directory / "report.json")
            if job["name"] in plan["reports_sha256"] and plan["reports_sha256"][job["name"]] != digest:
                raise ValueError("A completed job report changed")
            if phase != "benchmark":
                identities.add(report["sample_ids_sha256"])
            else:
                identities.add(bank_identity(directory, report))
            if len(identities) != 1:
                raise ValueError("Different sample identities across pipelines")
            plan["reports_sha256"][job["name"]] = digest
            if job["name"] not in plan["completed"]:
                plan["completed"].append(job["name"])
            write_json(output / "plan.json", plan)
        plan["status"] = "complete"
    except Exception as error:
        plan.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(output / "plan.json", plan)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("qualification", "full", "benchmark"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", help="Print jobs without accessing data or starting processes")
    parser.add_argument("--resume", action="store_true", help="Verify/reuse complete jobs; never overwrite partial results")
    parser.add_argument(
        "--new-campaign",
        type=Path,
        help="Qualification only: reuse prepared assets with the active Python; save config.json in a new results directory",
    )
    parser.add_argument("--onnx-python", type=Path, help="With --new-campaign: use a separate interpreter for ONNX jobs")
    args = parser.parse_args()
    if args.onnx_python and not args.new_campaign:
        parser.error("--onnx-python requires --new-campaign; subsequent phases use the saved config")
    if args.new_campaign and (args.phase != "qualification" or args.resume or args.dry_run):
        parser.error("--new-campaign requires qualification without --resume or --dry-run")
    config = configuration(args.config.resolve())
    if args.onnx_python:
        config["onnx_python"] = os.path.abspath(args.onnx_python.expanduser())
    if args.new_campaign:
        config = new_campaign(config, args.new_campaign)
    if args.dry_run:
        print(json.dumps(jobs(config, args.phase), indent=2))
    else:
        run(config, args.phase, args.resume)
