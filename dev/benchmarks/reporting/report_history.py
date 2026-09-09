"""Archive compact CPU/TensorRT comparisons and render a standalone HTML history."""

import hashlib
import html
import json
import math
import os
import re
import tempfile
from argparse import ArgumentParser
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

METRICS = ("f1", "recall", "precision", "coverage", "theilU")
METRIC_NAMES = dict(zip(METRICS, ("Macro-F1", "Macro-Recall", "Macro-Precision", "Coverage", "Theil U"), strict=True))
IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
REVISION = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
KINDS = ("tensorrt_deployment", "onnx_cpu_deployment")


def number(value):
    """Undefined metrics stay undefined; malformed/nonfinite values cannot be published."""
    if value is None:
        return None
    if type(value) not in (float, int) or not math.isfinite(value):
        raise ValueError("Expected a finite numeric value or null")
    return value


def quality_metrics(quality):
    result = []
    for index, level in enumerate(quality.get("levels", [])):
        item = {key: level[key] for key in ("name", "samples", "prediction_changes")}
        item["metrics"] = {}
        for metric in METRICS:
            a, b = [number(quality["models"][role]["metrics"][metric][str(index)]) for role in ("baseline", "candidate")]
            delta = number(level["candidate_minus_baseline"][metric])
            if a is None or b is None:
                if delta is not None:
                    raise ValueError("Undefined metrics require an undefined delta")
            elif delta is None or not math.isclose(delta, b - a, rel_tol=1e-10, abs_tol=1e-12):
                raise ValueError("Metric delta disagrees with baseline/candidate values")
            item["metrics"][metric] = {"baseline": a, "candidate": b, "delta": delta}
        result.append(item)
    return result


def project(source):
    """Select aggregate evidence without paths, predictions, labels, or exception text."""
    if "exit_code" in source:
        if type(source["exit_code"]) is not int or source["exit_code"] == 0:
            raise ValueError("A completed target command requires its evaluation report")
        return {
            "status": "failed",
            "phase": source["phase"],
            "exit_code": source["exit_code"],
            "engines": {},
            "quality": [],
            "resources": [],
        }
    if source.get("schema_version") != 1 or source.get("status") not in ("evaluated", "failed") or "builds" not in source:
        raise ValueError("Expected a version-one composed TensorRT deployment report")
    result = {"status": source["status"], "phase": source.get("phase"), "engines": {}, "quality": [], "resources": []}
    for role, build in source["builds"].items():
        if role not in ("baseline", "candidate"):
            raise ValueError("Unexpected model role")
        engine = build["engine"]
        result["engines"][role] = {key: engine[key] for key in ("sha256", "bytes", "context_memory_bytes")}
    result["quality"] = quality_metrics(source.get("quality", {}))
    for trial in source.get("trials", []):
        item = {"trial": trial["trial"], "memory": {}}
        if "latency" in trial:
            item["latency_ratio"] = number(trial["latency"]["median_paired_ratio"])
            item["latency_seconds"] = {role: number(trial["latency"]["median_seconds"][role]) for role in ("baseline", "candidate")}
        for role, snapshots in trial.get("memory", {}).items():
            item["memory"][role] = {
                name: {
                    "device_used_bytes": number(snapshots[name]["device_used_bytes"]),
                    "host_resident_bytes": number(snapshots[name]["host"]["resident_bytes"]),
                }
                for name in ("cuda_initialized", "warm")
            }
        result["resources"].append(item)
    if source["status"] == "evaluated":
        if set(result["engines"]) != {"baseline", "candidate"} or not result["quality"]:
            raise ValueError("Completed comparison requires both engines and quality")
        if len(result["resources"]) != source["settings"]["trials"] or not result["resources"]:
            raise ValueError("Completed comparison has incomplete resource trials")
        if any("latency_ratio" not in trial or set(trial["memory"]) != {"baseline", "candidate"} for trial in result["resources"]):
            raise ValueError("Completed comparison has incomplete resource measurements")
    return result


def runtime_metadata(source, root):
    """Use a hash-verified latency report for the actual measured runtime identity."""
    for stage in source.get("stages", []):
        name = stage["name"]
        if not name.startswith("latency-") or stage["status"] != "passed":
            continue
        if not IDENTIFIER.fullmatch(name):
            raise ValueError("Unsafe stage identifier")
        payload = (root / name / "report.json").read_bytes()
        if hashlib.sha256(payload).hexdigest() != stage["report_sha256"]:
            raise ValueError("Latency evidence changed since evaluation")
        child = json.loads(payload)
        return {
            "versions": {key: child["versions"][key] for key in ("torch", "tensorrt", "numpy")},
            "environment": {key: child["environment"][key] for key in ("gpu", "compute_capability", "platform", "python")},
        }
    if source.get("status") == "evaluated":
        raise ValueError("Completed comparison requires retained latency runtime evidence")
    return None


def cpu_evidence(source, root):
    """Read CPU measurements from their retained, hash-verified subprocess reports."""
    if source.get("schema_version") != 1 or source.get("status") not in ("evaluated", "failed"):
        raise ValueError("Expected a version-one composed CPU deployment report")
    stages = {stage["name"]: stage for stage in source["stages"]}
    if len(stages) != len(source["stages"]):
        raise ValueError("Duplicate CPU stages")

    def read(name, digest):
        payload = (root / name / "report.json").read_bytes()
        if hashlib.sha256(payload).hexdigest() != digest:
            raise ValueError("CPU evidence changed since evaluation")
        return json.loads(payload)

    result = {"status": source["status"], "phase": source.get("phase"), "models": {}, "execution": {}, "quality": [], "resources": []}
    comparison = {"required_candidate_ops": source["required_candidate_ops"]}
    runtime = None
    if "quality" in source:
        quality = read("quality", source["quality"]["report_sha256"])
        result["quality"] = quality_metrics(quality)
        if result["quality"] != quality_metrics(source["quality"]):
            raise ValueError("CPU quality summary disagrees with retained evidence")
        comparison["manifest_sha256"] = quality["manifest"]["sha256"]
    for role in source.get("execution", {}):
        if role not in ("baseline", "candidate"):
            raise ValueError("Unexpected CPU model role")
        name = f"placement-{role}"
        if stages[name]["status"] != "passed":
            raise ValueError("CPU placement did not finish")
        placement = read(name, stages[name]["report_sha256"])
        model = placement["models"][0]
        if model["execution"] != source["execution"][role]:
            raise ValueError("CPU placement summary disagrees with retained evidence")
        result["models"][role] = [{key: file[key] for key in ("sha256", "bytes")} for file in model["files"]]
        result["execution"][role] = [{key: op[key] for key in ("op", "provider", "count")} for op in model["execution"]]
        digest = placement["inputs"]["sha256"]
        if comparison.get("inputs_sha256", digest) != digest:
            raise ValueError("CPU placement inputs differ")
        comparison["inputs_sha256"] = digest
    for index, pair in enumerate(source["pairs"]):
        if type(pair["trial"]) is not int or pair["trial"] != index:
            raise ValueError("CPU trial indices must be consecutive")
        children = {}
        for role in ("baseline", "candidate"):
            name = f"trial-{index}-{role}"
            stage = stages[name]
            if stage["status"] != "measured":
                raise ValueError("CPU trial did not finish")
            child = read(name, stage["report_sha256"])
            if child["status"] != "measured":
                raise ValueError("CPU child did not finish")
            children[role] = child
            files = [{key: file[key] for key in ("sha256", "bytes")} for file in child["model_files"]]
            if role in result["models"] and files != result["models"][role]:
                raise ValueError("CPU model identity changed across trials")
            result["models"][role] = files
            metadata = {
                "versions": {key: child["versions"][key] for key in ("onnxruntime", "onnx", "numpy")},
                "environment": {key: child["environment"][key] for key in ("platform", "machine", "python", "cpu_affinity")},
                "runtime_build": child["runtime_build"],
            }
            if runtime is not None and runtime != metadata:
                raise ValueError("CPU runtime changed across trials")
            runtime = metadata
            settings = {key: child["settings"][key] for key in ("threads", "inter_op_threads", "warmup", "repeats", "optimization")}
            for key, value in (("settings", settings), ("inputs_sha256", child["inputs"]["sha256"])):
                if key in comparison and comparison[key] != value:
                    raise ValueError("CPU settings or inputs changed across trials")
                comparison[key] = value
        latency = {role: number(child["median_seconds"]) for role, child in children.items()}
        memory = {
            role: {key: number(child["memory"]["after_measurement"][key]) for key in ("resident_bytes", "peak_resident_bytes")}
            for role, child in children.items()
        }
        for key, numerator, denominator in (
            ("warm_latency", latency["candidate"], latency["baseline"]),
            *((key, memory["candidate"][key], memory["baseline"][key]) for key in ("resident_bytes", "peak_resident_bytes")),
        ):
            ratio = number(pair["candidate_over_baseline"][key])
            if denominator is None or denominator <= 0 or numerator is None or ratio is None:
                raise ValueError("CPU resource measurements must have defined ratios")
            if not math.isclose(ratio, numerator / denominator, rel_tol=1e-10, abs_tol=1e-12):
                raise ValueError("CPU resource ratio disagrees with retained evidence")
        result["resources"].append(
            {"trial": index, "latency_seconds": latency, "latency_ratio": pair["candidate_over_baseline"]["warm_latency"], "memory": memory}
        )
    if source["status"] == "evaluated":
        if not result["quality"] or not result["resources"] or set(result["execution"]) != {"baseline", "candidate"}:
            raise ValueError("Completed CPU evaluation requires quality, placement and resource trials")
    comparison["recorded_trials"] = len(result["resources"])
    if "settings" in source:
        requested = {key: source["settings"][key] for key in ("threads", "trials", "warmup", "repeats")}
        if any(type(value) is not int or value < 1 for value in requested.values()):
            raise ValueError("Requested CPU settings must be positive integers")
        if "settings" in comparison and any(comparison["settings"][key] != requested[key] for key in ("threads", "warmup", "repeats")):
            raise ValueError("Observed CPU settings disagree with requested settings")
        if comparison["recorded_trials"] > requested["trials"] or (
            source["status"] == "evaluated" and comparison["recorded_trials"] != requested["trials"]
        ):
            raise ValueError("CPU trial count disagrees with requested budget")
        comparison["requested_settings"] = requested
    return result, runtime, comparison


def archive(report, history, run_id, revision, profile, run_url=None, performance_valid=False, note=""):
    if not IDENTIFIER.fullmatch(run_id) or not REVISION.fullmatch(revision):
        raise ValueError("Require a safe run identifier and full hexadecimal source revision")
    if not profile.strip():
        raise ValueError("Supply a descriptive profile")
    if run_url is not None:
        url = urlsplit(run_url)
        if url.scheme != "https" or not url.netloc or url.username or url.password:
            raise ValueError("Run URL must be HTTPS without credentials")
    payload = Path(report).read_bytes()
    source = json.loads(payload)
    cpu = "pairs" in source and "required_candidate_ops" in source and "builds" not in source
    if cpu:
        evidence, hardware, comparison = cpu_evidence(source, Path(report).parent)
    else:
        evidence = project(source)
        hardware = runtime_metadata(source, Path(report).parent)
    record = {
        "schema_version": 1,
        "kind": "onnx_cpu_deployment" if cpu else "tensorrt_deployment",
        "run_id": run_id,
        "revision": revision,
        "profile": profile,
        "run_url": run_url,
        "note": note,
        "source_report_sha256": hashlib.sha256(payload).hexdigest(),
        "runner_sha256": source.get("runner_sha256"),
        "performance_valid": bool(performance_valid and evidence["status"] == "evaluated"),
        "evidence": evidence,
        "runtime": hardware,
        "comparison": {
            "inputs_sha256": source.get("inputs", {}).get("sha256"),
            "manifest_sha256": source.get("manifest", {}).get("sha256"),
            "settings": {
                key: value
                for key, value in source.get("settings", {}).items()
                if key in ("trials", "warmup", "repeats", "memory_runs", "threads", "device", "pinned_host_io")
            },
            "build_settings": {
                role: {
                    key: value
                    for key, value in build.get("settings", {}).items()
                    if key in ("profiles", "fp16_allowed", "tf32_allowed", "workspace_bytes", "builder_optimization")
                }
                for role, build in source.get("builds", {}).items()
            },
        },
    }
    if cpu:
        record["comparison"] = comparison
    records = Path(history) / "records"
    records.mkdir(parents=True, exist_ok=True)
    destination = records / f"{run_id}.json"
    if destination.exists():
        old = json.loads(destination.read_text())
        record["recorded_at"] = old["recorded_at"]
        if old != record:
            raise ValueError("Run identity already exists with different evidence or metadata")
        return destination
    record["recorded_at"] = datetime.now(UTC).isoformat()
    encoded = (json.dumps(record, indent=2, allow_nan=False) + "\n").encode()
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=records, suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        # Publish without overwriting a concurrent writer's immutable record.
        os.link(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination


def escape(value):
    return html.escape(str(value), quote=True)


def display(value, scale=1):
    return "undefined" if value is None else f"{number(value) * scale:.4f}"


def render(history):
    history = Path(history)
    records = []
    for path in sorted((history / "records").glob("*.json")):
        record = json.loads(path.read_text())
        if record.get("schema_version") != 1 or record.get("kind") not in KINDS:
            raise ValueError(f"Unsupported history record: {path.name}")
        if not IDENTIFIER.fullmatch(record["run_id"]) or path.name != record["run_id"] + ".json":
            raise ValueError("History filename and run identity differ")
        records.append(record)
    lines = [
        '<!doctype html><html lang="en"><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        "<title>Quantization evaluation history</title>",
        "<style>body{font:16px system-ui;max-width:1100px;margin:2rem auto;padding:0 1rem;color:#192331;background:#f5f7fa}"
        "article{background:white;border:1px solid #cbd5e1;border-radius:8px;padding:1.2rem;margin:1rem 0;overflow:auto}"
        "table{border-collapse:collapse;width:100%;margin:1rem 0}th,td{text-align:left;padding:.5rem;border-bottom:1px solid #ddd}"
        "code{overflow-wrap:anywhere}.failed{border-left:5px solid #a32020}.note{color:#46556a}</style>",
        "<h1>Quantization evaluation history</h1>",
        "<p>Completed evaluation is not production acceptance. Compare quality and resource use on the intended hardware. "
        "Device memory readings are device-wide snapshots, not per-process or transient peaks.</p>",
    ]
    for record in sorted(records, key=lambda r: (r["recorded_at"], r["run_id"]), reverse=True):
        evidence = record["evidence"]
        lines.append(f'<article class="{"failed" if evidence["status"] == "failed" else "evaluated"}">')
        lines.append(f"<h2>{escape(record['profile'])} — {escape(record['run_id'])}</h2>")
        lines.append(
            f"<p>Status: <strong>{escape(evidence['status'])}</strong>; phase: {escape(evidence['phase'])}. "
            f"Recorded: {escape(record['recorded_at'])}.</p>"
        )
        lines.append(
            f"<p>Revision <code>{escape(record['revision'])}</code>. "
            f'<a href="records/{escape(record["run_id"])}.json">Download compact record</a></p>'
        )
        if record.get("run_url"):
            url = urlsplit(record["run_url"])
            if url.scheme != "https" or not url.netloc or url.username or url.password:
                raise ValueError("Unsafe history run URL")
            lines.append(f'<p><a href="{escape(record["run_url"])}">Original workflow run</a></p>')
        lines.append(f'<p class="note">{escape(record["note"])}</p>')
        hardware = record.get("runtime")
        cpu = record["kind"] == "onnx_cpu_deployment"
        device = hardware["environment"]["machine" if cpu else "gpu"] if hardware else "unrecorded"
        lines.append(
            f"<p>Measured {'CPU architecture' if cpu else 'GPU'}: {escape(device)}. "
            f"Source report <code>{escape(record['source_report_sha256'])}</code>.</p>"
        )
        if cpu:
            requested = record["comparison"].get("requested_settings")
            count = record["comparison"]["recorded_trials"]
            budget = f"of {requested['trials']} requested" if requested else "(requested count was not retained)"
            lines.append(f"<p>Recorded CPU trials: {escape(count)} {escape(budget)}.</p>")
        for level in evidence["quality"]:
            lines.append(
                f"<h3>{escape(level['name'])}: {escape(level['samples'])} samples, "
                f"{escape(level['prediction_changes'])} changed predictions</h3>"
            )
            lines.append("<table><tr><th>Metric</th><th>Baseline</th><th>Candidate</th><th>Delta ×100</th></tr>")
            for metric in METRICS:
                values = level["metrics"][metric]
                lines.append(
                    f"<tr><td>{escape(METRIC_NAMES[metric])}</td><td>{display(values['baseline'])}</td>"
                    f"<td>{display(values['candidate'])}</td><td>{display(values['delta'], 100)}</td></tr>"
                )
            lines.append("</table>")
        if record["performance_valid"] and evidence["status"] == "evaluated" and cpu:
            lines.append(
                "<p>Resource measurements marked usable by the publisher; this is not independently certified. "
                "CPU latency compares separate-process medians and excludes image decoding/preprocessing. "
                "RSS includes process overhead; peak RSS is approximate.</p>"
                "<table><tr><th>Trial</th><th>CPU median latency ratio</th><th>Baseline RSS MiB</th>"
                "<th>Candidate RSS MiB</th><th>Baseline approximate peak MiB</th><th>Candidate approximate peak MiB</th></tr>"
            )
            for trial in evidence["resources"]:
                values = [str(trial["trial"]), display(trial["latency_ratio"])]
                values += [
                    display(trial["memory"][role][key], 1 / 2**20)
                    for key in ("resident_bytes", "peak_resident_bytes")
                    for role in ("baseline", "candidate")
                ]
                lines.append("<tr>" + "".join(f"<td>{escape(value)}</td>" for value in values) + "</tr>")
            lines.append("</table><p>Ratios are candidate / baseline; below one is lower.</p>")
        elif record["performance_valid"] and evidence["status"] == "evaluated":
            lines.append("<p>Resource measurements marked usable by the publisher; this is not independently certified.</p>")
            lines.append(
                "<table><tr><th>Trial</th><th>Paired latency ratio</th><th>Baseline device MiB</th>"
                "<th>Candidate device MiB</th><th>Baseline host RSS MiB</th><th>Candidate host RSS MiB</th></tr>"
            )
            for trial in evidence["resources"]:
                a, b = [trial["memory"][role]["warm"] for role in ("baseline", "candidate")]
                values = [
                    trial["trial"],
                    display(trial["latency_ratio"]),
                    display(a["device_used_bytes"], 1 / 2**20),
                    display(b["device_used_bytes"], 1 / 2**20),
                    display(a["host_resident_bytes"], 1 / 2**20),
                    display(b["host_resident_bytes"], 1 / 2**20),
                ]
                lines.append("<tr>" + "".join(f"<td>{escape(value)}</td>" for value in values) + "</tr>")
            lines.append(
                "</table><p>Latency ratios are candidate / baseline; below one is lower. "
                "Inspect initialization snapshots in the compact record.</p>"
            )
        else:
            lines.append("<p><strong>Resource readings excluded from performance comparisons.</strong></p>")
        if not evidence["quality"]:
            lines.append("<p>Quality results unavailable.</p>")
        lines.append("</article>")
    if not records:
        lines.append("<p>No archived runs.</p>")
    lines.append("</html>")
    history.mkdir(parents=True, exist_ok=True)
    (history / "index.html").write_text("\n".join(lines) + "\n")
    return history / "index.html"


def main():
    parser = ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    add = sub.add_parser("archive")
    for name in ("report", "history", "run-id", "revision", "profile"):
        add.add_argument("--" + name, required=True)
    add.add_argument("--run-url")
    add.add_argument("--performance-valid", action="store_true")
    add.add_argument("--note", default="")
    sub.add_parser("render").add_argument("--history", required=True)
    args = vars(parser.parse_args())
    command = args.pop("command")
    archive(**args) if command == "archive" else render(**args)


if __name__ == "__main__":
    main()
