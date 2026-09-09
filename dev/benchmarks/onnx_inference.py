"""Measure ONNX Session.run with explicit providers and portable preprocessed inputs."""

import hashlib
import json
import platform
import statistics
import time
from argparse import ArgumentParser
from collections import Counter
from pathlib import Path

import numpy as np


def file_hash(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def model_files(path, onnx):
    """Include external tensor storage, also inside graph attributes/subgraphs."""
    paths = {path.resolve()}

    def visit(message):
        if isinstance(message, onnx.TensorProto):
            for entry in message.external_data:
                if entry.key == "location":
                    paths.add((path.parent / entry.value).resolve())
        for field, value in message.ListFields():
            if field.message_type is not None:
                for child in value if field.is_repeated else [value]:
                    visit(child)

    visit(onnx.load(path, load_external_data=False))
    return [{"path": str(p), "sha256": file_hash(p), "bytes": p.stat().st_size} for p in sorted(paths)]


def run(models, inputs, output, provider, threads=1, warmup=3, repeats=11, provider_options=None):
    import onnx
    import onnxruntime as ort

    if min(threads, warmup, repeats) < 1:
        raise ValueError("threads, warmup and repeats must be positive")
    if not models or len(set(models)) != len(models):
        raise ValueError("Supply distinct model paths")
    if provider not in ort.get_available_providers():
        raise ValueError(f"Requested provider {provider} unavailable; available: {ort.get_available_providers()}")
    with np.load(inputs, allow_pickle=False) as archive:
        feeds = {name: np.ascontiguousarray(archive[name]) for name in archive.files}
    if not feeds or any(not np.isfinite(value).all() for value in feeds.values()):
        raise ValueError("Inputs must contain finite named arrays")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": 1,
        "status": "running",
        "environment": {"platform": platform.platform(), "machine": platform.machine(), "python": platform.python_version()},
        "versions": {"onnx": onnx.__version__, "onnxruntime": ort.__version__, "numpy": np.__version__},
        "runtime_build": ort.get_build_info(),
        "runner_sha256": file_hash(__file__),
        "provider": provider,
        "provider_options": provider_options or {},
        "available_providers": ort.get_available_providers(),
        "threads": threads,
        "warmup": warmup,
        "repeats": repeats,
        "inputs": {"path": str(inputs), "sha256": file_hash(inputs)},
        "input_arrays": {name: {"shape": list(a.shape), "dtype": str(a.dtype)} for name, a in feeds.items()},
        "scope": (
            "Warm Session.run with CPU NumPy inputs and outputs; includes device transfers, "
            "excludes image IO, preprocessing and session construction."
        ),
        "order": "alternating forward/reverse model order per repetition",
        "models": [],
    }
    sessions = []
    providers = [(provider, provider_options or {})]
    if provider != "CPUExecutionProvider":
        providers.append("CPUExecutionProvider")

    def options():
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        return opts

    try:
        for index, path in enumerate(map(Path, models)):
            record = {"path": str(path), "files": model_files(path, onnx), "seconds": []}
            report["models"].append(record)
            opts = options()
            opts.enable_profiling = True
            opts.profile_file_prefix = str(output / f"model-{index}-profile")
            session = ort.InferenceSession(str(path), sess_options=opts, providers=providers)
            session.disable_fallback()
            try:
                if set(feeds) != {node.name for node in session.get_inputs()}:
                    raise ValueError(f"Input names do not match {path}")
                predictions = session.run(None, feeds)
            finally:
                profile = session.end_profiling()
            counts = Counter(
                (event["args"].get("op_name"), event["args"].get("provider"))
                for event in json.loads(Path(profile).read_text())
                if event.get("cat") == "Node" and event.get("args", {}).get("provider")
            )
            record["execution"] = [{"op": op, "provider": ep, "count": n} for (op, ep), n in sorted(counts.items())]
            record["profile"] = str(profile)
            if not any(ep == provider for _, ep in counts):
                raise RuntimeError(f"No profiled operation executed on requested provider {provider}")
            if any(not np.isfinite(array).all() for array in predictions):
                raise ValueError(f"Nonfinite predictions from {path}")
            record["outputs"] = [node.name for node in session.get_outputs()]
            np.savez(output / f"model-{index}-outputs.npz", **dict(zip(record["outputs"], predictions, strict=True)))
            del session
            session = ort.InferenceSession(str(path), sess_options=options(), providers=providers)
            session.disable_fallback()
            record["session_providers"] = session.get_providers()
            record["session_provider_options"] = session.get_provider_options()
            sessions.append(session)
        for trial in range(warmup + repeats):
            order = range(len(sessions)) if trial % 2 else reversed(range(len(sessions)))
            for index in order:
                started = time.perf_counter()
                sessions[index].run(None, feeds)
                elapsed = time.perf_counter() - started
                if trial >= warmup:
                    report["models"][index]["seconds"].append(elapsed)
        for record in report["models"]:
            record["median_seconds"] = statistics.median(record["seconds"])
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, action="append", required=True)
    parser.add_argument("--inputs", type=Path, required=True, help="NPZ with exact ONNX input names and preprocessed arrays")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--provider-options", type=json.loads, default={})
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=11)
    args = vars(parser.parse_args())
    args["models"] = args.pop("model")
    run(**args)


if __name__ == "__main__":
    main()
