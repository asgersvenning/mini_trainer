"""Logged subprocess stages shared by the inference comparison orchestrators."""

import json
import os
import subprocess
from pathlib import Path

from .onnx_inference import file_hash


def run_stage(output, report, name, command, expected, *, env=None):
    """Persist before launching; the caller owns final reporting and failure cleanup."""
    stage = {"name": name, "command": command, "status": "running", "log": f"{name}.log"}
    report["stages"].append(stage)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    with (output / stage["log"]).open("w") as log:
        process = subprocess.run(
            command,
            cwd=Path(__file__).resolve().parents[3],
            env={**os.environ, "PYTHONHASHSEED": "0", **(env or {})},
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    stage["returncode"] = process.returncode
    if process.returncode:
        raise RuntimeError(f"{name} failed with exit code {process.returncode}; see {output / stage['log']}")
    path = output / name / "report.json"
    result = json.loads(path.read_text())
    if result["status"] != expected:
        raise RuntimeError(f"Unexpected {name} status: {result['status']}")
    stage.update(status=expected, report_sha256=file_hash(path))
    return result
