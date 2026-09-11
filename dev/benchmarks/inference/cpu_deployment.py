"""Compose CPU deployment quality, placement and isolated resource measurements."""

import json
import os
import subprocess
import sys
from argparse import ArgumentParser
from pathlib import Path

from .inference_pair import run_pair
from .onnx_inference import file_hash


def evaluate(
    baseline, candidate, manifest, inputs, output, threads=1, trials=3, warmup=3, repeats=31, required_ops=(), metrics_python=sys.executable
):
    if min(threads, trials, warmup, repeats) < 1:
        raise ValueError("Threads, trials, warmup and repeats must be positive")
    paths = {name: Path(path).resolve(strict=True) for name, path in (("baseline", baseline), ("candidate", candidate))}
    manifest, inputs, output = Path(manifest).resolve(strict=True), Path(inputs).resolve(strict=True), Path(output).resolve()
    inputs_hash = file_hash(inputs)
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": 1,
        "status": "running",
        "runner_sha256": file_hash(__file__),
        "settings": {"threads": threads, "trials": trials, "warmup": warmup, "repeats": repeats},
        "required_candidate_ops": list(required_ops),
        "stages": [],
        "pairs": [],
        "scope": (
            "CPU quality, observed operation placement and separate-process resource trials; no automatic acceptance gate. "
            "Latency excludes image decoding/preprocessing. Peak RSS is approximate and includes process overhead. "
            "Target claims require execution on that target; no thermal control or disk-cache eviction."
        ),
    }

    def save():
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    def child(name, module, arguments, expected):
        command = [sys.executable, "-m", f"dev.benchmarks.inference.{module}", *map(str, arguments), "--output", str(output / name)]
        stage = {"name": name, "command": command, "status": "running", "log": f"{name}.log"}
        report["stages"].append(stage)
        save()
        with (output / stage["log"]).open("w") as log:
            process = subprocess.run(
                command,
                cwd=Path(__file__).resolve().parents[3],
                env={**os.environ, "PYTHONHASHSEED": "0"},
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        stage["returncode"] = process.returncode
        if process.returncode:
            raise RuntimeError(f"{name} failed; see {output / stage['log']}")
        result_path = output / name / "report.json"
        result = json.loads(result_path.read_text())
        if result["status"] != expected:
            raise RuntimeError(f"Unexpected {name} status: {result['status']}")
        stage.update(status=expected, report_sha256=file_hash(result_path))
        return result

    try:
        report["phase"] = "quality"
        save()
        runtime = {"backend": "onnx", "threads": threads}
        quality = run_pair(manifest, paths["baseline"], paths["candidate"], output / "quality", runtime, runtime, metrics_python)
        report["quality"] = {
            "report_sha256": file_hash(output / "quality/report.json"),
            "levels": quality["levels"],
            "models": quality["models"],
            "undefined_metrics": quality["undefined_metrics"],
        }
        expected_files = {role: json.loads((output / f"quality/{role}/report.json").read_text())["model_files"] for role in paths}
        report["phase"] = "placement"
        report["execution"] = {}
        for role, model in paths.items():
            args = [
                "--model",
                model,
                "--inputs",
                inputs,
                "--provider",
                "CPUExecutionProvider",
                "--threads",
                threads,
                "--warmup",
                1,
                "--repeats",
                1,
            ]
            if role == "candidate":
                for operation in required_ops:
                    args.extend(["--require-provider-op", operation])
            placement = child(f"placement-{role}", "onnx_inference", args, "passed")
            if placement["models"][0]["files"] != expected_files[role] or placement["inputs"]["sha256"] != inputs_hash:
                raise ValueError("Model or timing inputs changed between quality and placement")
            report["execution"][role] = placement["models"][0]["execution"]
        report["phase"] = "resources"
        for trial in range(trials):
            pair = {}
            order = ("baseline", "candidate") if trial % 2 == 0 else ("candidate", "baseline")
            for role in order:
                result = child(
                    f"trial-{trial}-{role}",
                    "onnx_cpu_memory",
                    ["--model", paths[role], "--inputs", inputs, "--threads", threads, "--warmup", warmup, "--repeats", repeats],
                    "measured",
                )
                if result["model_files"] != expected_files[role] or result["inputs"]["sha256"] != inputs_hash:
                    raise ValueError("Model or timing inputs changed between quality and resource measurement")
                pair[role] = result
            a, b = pair["baseline"], pair["candidate"]
            if any(a[key] != b[key] for key in ("settings", "versions", "environment", "runtime_build", "memory_sources")):
                raise ValueError("Paired resource environments or settings differ")
            ratios = {"warm_latency": b["median_seconds"] / a["median_seconds"]}
            for key in ("resident_bytes", "peak_resident_bytes"):
                ratios[key] = b["memory"]["after_measurement"][key] / a["memory"]["after_measurement"][key]
            report["pairs"].append({"trial": trial, "order": list(order), "candidate_over_baseline": ratios})
            save()
        lines = [
            "# CPU deployment comparison",
            "",
            report["scope"],
            "",
            "| Trial | Warm latency ratio | Resident memory ratio | Approximate peak ratio |",
            "| --- | ---: | ---: | ---: |",
        ]
        for pair in report["pairs"]:
            ratios = pair["candidate_over_baseline"]
            lines.append(
                f"| {pair['trial']} | {ratios['warm_latency']:.4f} | {ratios['resident_bytes']:.4f} | {ratios['peak_resident_bytes']:.4f} |"
            )
        lines.extend(
            [
                "",
                "Ratios are candidate / baseline; below one is lower. "
                "These compare separate-process medians, not adjacent inference pairs.",
                "",
                (output / "quality/summary.md").read_text(),
            ]
        )
        (output / "summary.md").write_text("\n".join(lines))
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
    for name in ("baseline", "candidate", "manifest", "inputs", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name, default in (("threads", 1), ("trials", 3), ("warmup", 3), ("repeats", 31)):
        parser.add_argument("--" + name, type=int, default=default)
    parser.add_argument(
        "--require-provider-op",
        dest="required_ops",
        action="append",
        default=[],
        help="Required candidate CPU operation type; inspect counts for coverage",
    )
    parser.add_argument("--metrics-python", default=sys.executable)
    evaluate(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
