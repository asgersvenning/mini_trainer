"""Run paired held-out inference and mini_metrics evaluation in fresh processes."""

import json
import os
import subprocess
import sys
from argparse import ArgumentParser
from pathlib import Path

from .onnx_inference import file_hash


def run_pair(manifest, baseline, candidate, output, baseline_runtime=None, candidate_runtime=None, metrics_python=sys.executable):
    """Compose maintained collectors without loading either runtime in this process.

    Runtime dictionaries accept a Python executable and collector options. Paths
    are resolved from the caller's directory; child commands run from the checkout.
    No dependencies are installed and existing output directories are never reused.
    """
    manifest = Path(manifest).resolve(strict=True)
    models = [Path(path).resolve(strict=True) for path in (baseline, candidate)]
    commands = []
    output = Path(output).resolve()
    for name, model, runtime in zip(("baseline", "candidate"), models, (baseline_runtime, candidate_runtime), strict=True):
        options = dict(runtime or {})
        python = str(Path(options.pop("python", sys.executable)).absolute())
        allowed = {"backend", "provider", "provider_options", "threads", "optimization", "device", "save_scores"}
        if options.keys() - allowed:
            raise ValueError(f"Unknown {name} runtime options: {sorted(options.keys() - allowed)}")
        command = [
            python,
            "-m",
            "dev.benchmarks.dataset_inference",
            "--model",
            str(model),
            "--manifest",
            str(manifest),
            "--output",
            str(output / name),
        ]
        for key, value in options.items():
            if key == "save_scores":
                if not isinstance(value, bool):
                    raise ValueError("save_scores must be a boolean")
                if value:
                    command.append("--save-scores")
            else:
                command.extend(["--" + key.replace("_", "-"), json.dumps(value) if key == "provider_options" else str(value)])
        if name == "candidate":
            command.extend(["--baseline-bundle", str(output / "baseline/evaluation.json")])
        commands.append((name, command, "inferred"))
    commands.append(
        (
            "quality",
            [
                str(Path(metrics_python).absolute()),
                "-m",
                "dev.benchmarks.quality_compare",
                "--manifest",
                str(output / "candidate/comparison.json"),
                "--output",
                str(output / "quality"),
            ],
            "evaluated",
        )
    )
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": 1,
        "status": "running",
        "runner_sha256": file_hash(__file__),
        "manifest": {"path": str(manifest), "sha256": file_hash(manifest)},
        "python_hash_seed": "0",
        "stages": [],
        "scope": "Paired held-out quality only; not timing, integer placement or production acceptance.",
    }
    report_path = output / "report.json"

    def save():
        report_path.write_text(json.dumps(report, indent=2) + "\n")

    try:
        for name, command, expected in commands:
            stage = {"name": name, "command": command, "status": "running", "log": f"{name}.log"}
            report["stages"].append(stage)
            save()
            with (output / stage["log"]).open("w") as log:
                result = subprocess.run(
                    command,
                    cwd=Path(__file__).resolve().parents[2],
                    env={**os.environ, "PYTHONHASHSEED": "0"},
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            stage["returncode"] = result.returncode
            if result.returncode:
                stage["status"] = "failed"
                raise RuntimeError(f"{name} failed with exit code {result.returncode}; see {output / stage['log']}")
            child_path = output / name / "report.json"
            child = json.loads(child_path.read_text())
            if child["status"] != expected:
                raise RuntimeError(f"Unexpected {name} report status: {child['status']}")
            stage.update(status=expected, report_sha256=file_hash(child_path))
            save()
        report.update(status="evaluated", levels=child["levels"], models=child["models"], undefined_metrics=child["undefined_metrics"])
        lines = [
            "# Paired inference quality",
            "",
            report["scope"],
            "",
            "| Level | Samples | Changed predictions | F1 delta | Recall delta | Precision delta | Coverage delta | Theil U delta |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for level in child["levels"]:
            values = [level["candidate_minus_baseline"][metric] for metric in ("f1", "recall", "precision", "coverage", "theilU")]
            label = str(level["name"]).replace("|", "\\|").replace("\n", " ").replace("\r", " ")
            lines.append(
                "| "
                + " | ".join(
                    [
                        label,
                        str(level["samples"]),
                        str(level["prediction_changes"]),
                        *["undefined" if v is None else f"{v:+.6f}" for v in values],
                    ]
                )
                + " |"
            )
        lines.extend(["", "Deltas are candidate minus baseline in metric units. Undefined values are not passes.", ""])
        (output / "summary.md").write_text("\n".join(lines))
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
    for name in ("manifest", "baseline", "candidate", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("baseline", "candidate"):
        parser.add_argument(f"--{name}-runtime", type=json.loads, help="JSON collector options and optional Python executable path")
    parser.add_argument("--metrics-python", default=sys.executable, help="Explicit evaluation environment Python; installs nothing")
    run_pair(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
