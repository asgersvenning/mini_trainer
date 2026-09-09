import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture
def environment(tmp_path):
    executable = tmp_path / "prepared python"
    executable.write_text(
        f"#!{sys.executable}\n"
        + """
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
with Path(os.environ['HARNESS_COMMANDS']).open('a') as log:
    log.write(json.dumps(args) + '\\n')
if args[0] == '-c':
    stage = 'metrics' if 'mini_metrics' in args[1] else 'preflight'
else:
    stage = 'candidate' if '--model' in args and args[args.index('--model')+1] == os.environ['TRT_CANDIDATE_MODEL'] else 'baseline'
    if args[1].endswith('tensorrt_deployment'): stage = 'evaluation'
if stage == os.environ.get('FAIL_STAGE'):
    print('requested stage failure', file=sys.stderr)
    raise SystemExit(19)
"""
    )
    executable.chmod(0o755)
    env = {
        **os.environ,
        "BENCHMARK_PYTHON": str(executable),
        "BENCHMARK_METRICS_PYTHON": str(executable),
        "TRT_BASELINE_MODEL": "float model.onnx",
        "TRT_CANDIDATE_MODEL": "candidate $(not-a-command).onnx",
        "TRT_INFERENCE_MANIFEST": "held out/manifest.json",
        "TRT_INPUTS": "held out/inputs.npz",
        "TRT_PROFILES": "profiles with spaces.json",
        "HARNESS_COMMANDS": str(tmp_path / "commands.jsonl"),
    }
    return env


def invoke(tmp_path, environment):
    output = tmp_path / "new results"
    result = subprocess.run(["bash", "dev/check-tensorrt-deployment.sh", str(output)], env=environment, capture_output=True, text=True)
    commands = Path(environment["HARNESS_COMMANDS"])
    calls = [json.loads(line) for line in commands.read_text().splitlines()] if commands.exists() else []
    return result, output, calls


def test_harness_builds_before_evaluation_and_preserves_literal_paths(tmp_path, environment):
    result, output, calls = invoke(tmp_path, environment)
    assert result.returncode == 0, result.stderr
    assert len(calls) == 5
    assert [call[1] for call in calls[2:]] == ["dev.benchmarks.inference.tensorrt_build"] * 2 + [
        "dev.benchmarks.inference.tensorrt_deployment"
    ]
    for role, call in zip(("BASELINE", "CANDIDATE"), calls[2:4], strict=True):
        assert call[call.index("--model") + 1] == environment[f"TRT_{role}_MODEL"]
        assert call[call.index("--profiles") + 1] == environment["TRT_PROFILES"]
        assert "--fp16" in call and "--tf32" not in call
    assert calls[4][calls[4].index("--metrics-python") + 1] == environment["BENCHMARK_METRICS_PYTHON"]
    assert json.loads((output / "status.json").read_text()) == {"phase": "complete", "exit_code": 0}
    assert (output / "revision.txt").read_text().strip()
    original = (output / "status.json").read_bytes()
    again, _, repeated = invoke(tmp_path, environment)
    assert again.returncode == 2 and repeated == calls and (output / "status.json").read_bytes() == original


@pytest.mark.parametrize(
    "stage,phase,count",
    [("preflight", "preflight", 1), ("metrics", "preflight", 2), ("candidate", "build-candidate", 4), ("evaluation", "evaluation", 5)],
)
def test_harness_retains_failed_phase_and_stops(tmp_path, environment, stage, phase, count):
    environment["FAIL_STAGE"] = stage
    result, output, calls = invoke(tmp_path, environment)
    assert result.returncode == 19 and len(calls) == count
    assert json.loads((output / "status.json").read_text()) == {"phase": phase, "exit_code": 19}
    assert any("requested stage failure" in path.read_text() for path in output.glob("*.log"))


def test_missing_configuration_stops_before_importing_runtimes(tmp_path, environment):
    del environment["TRT_PROFILES"]
    result, output, calls = invoke(tmp_path, environment)
    assert result.returncode != 0 and not calls
    assert json.loads((output / "status.json").read_text())["phase"] == "configuration"
    assert "Set TRT_PROFILES" in result.stderr
