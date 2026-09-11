"""Measure one ONNX CPU model per fresh Linux process, without profiling."""

import json
import os
import platform
import statistics
import time
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from .onnx_inference import file_hash, model_files


def resident_memory():
    """Linux mapping snapshot and approximate post-exec high-water mark, in bytes."""
    if platform.system() != "Linux":
        raise RuntimeError("This memory probe requires Linux /proc")
    fields = {}
    for path, names in (("status", ("VmRSS", "VmHWM")), ("smaps_rollup", ("Rss", "Pss", "Swap"))):
        for line in Path(f"/proc/self/{path}").read_text().splitlines():
            name, _, value = line.partition(":")
            if name in names:
                amount, unit = value.split()
                if unit != "kB":
                    raise RuntimeError(f"Unexpected /proc memory unit: {unit}")
                fields[name] = int(amount) * 1024
    return {
        "resident_bytes": fields["Rss"],
        "proportional_resident_bytes": fields["Pss"],
        "swap_bytes": fields["Swap"],
        "approximate_status_resident_bytes": fields["VmRSS"],
        "peak_resident_bytes": fields["VmHWM"],
    }


def measure(model, inputs, output, threads=1, warmup=3, repeats=31, optimization="all"):
    """Use the CLI in a fresh process: in-process calls inherit earlier RSS peaks."""
    if min(threads, warmup, repeats) < 1 or optimization not in ("all", "disable"):
        raise ValueError("Positive threads/warmup/repeats and all/disable optimization are required")
    model, inputs = Path(model).resolve(strict=True), Path(inputs).resolve(strict=True)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": 1,
        "status": "running",
        "runner_sha256": file_hash(__file__),
        "environment": {"platform": platform.platform(), "machine": platform.machine(), "python": platform.python_version()},
        "settings": {"threads": threads, "inter_op_threads": 1, "warmup": warmup, "repeats": repeats, "optimization": optimization},
        "memory": {},
        "memory_sources": {
            "resident_bytes": "/proc/self/smaps_rollup Rss",
            "proportional_resident_bytes": "/proc/self/smaps_rollup Pss",
            "swap_bytes": "/proc/self/smaps_rollup Swap",
            "peak_resident_bytes": "/proc/self/status VmHWM (approximate)",
        },
        "scope": (
            "Single CPU session, preprocessed NumPy IO, no profiling. RSS includes interpreter, imports, inputs and outputs. "
            "Peak is the approximate high-water mark since exec through each snapshot, not model-only memory. "
            "Memory includes finite-input/output validation. Snapshot reads occur outside inference timings. "
            "Run in a fresh process for each model/trial. Snapshots precede ONNX artifact inspection and output serialization. "
            "Session load is not guaranteed disk-cold; inference excludes image decoding and preprocessing."
        ),
    }
    try:
        report["memory"]["before_runtime_import"] = resident_memory()
        import onnxruntime as ort

        report["versions"] = {"onnxruntime": ort.__version__, "numpy": np.__version__}
        report["runtime_build"] = ort.get_build_info()
        report["environment"]["cpu_affinity"] = sorted(os.sched_getaffinity(0))
        report["memory"]["after_runtime_import"] = resident_memory()
        with np.load(inputs, allow_pickle=False) as archive:
            feeds = {name: np.ascontiguousarray(archive[name]) for name in archive.files}
        if not feeds or any(not np.isfinite(value).all() for value in feeds.values()):
            raise ValueError("Inputs must contain finite named arrays")
        report["input_arrays"] = {name: {"shape": list(a.shape), "dtype": str(a.dtype)} for name, a in feeds.items()}
        report["memory"]["before_session"] = resident_memory()
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.graph_optimization_level = getattr(
            ort.GraphOptimizationLevel, "ORT_ENABLE_ALL" if optimization == "all" else "ORT_DISABLE_ALL"
        )
        started = time.perf_counter()
        session = ort.InferenceSession(str(model), sess_options=options, providers=["CPUExecutionProvider"])
        report["session_load_seconds"] = time.perf_counter() - started
        session.disable_fallback()
        report["session_providers"] = session.get_providers()
        report["memory"]["after_session"] = resident_memory()
        if report["session_providers"] != ["CPUExecutionProvider"]:
            raise RuntimeError("Expected an exclusively CPU session")
        if set(feeds) != {node.name for node in session.get_inputs()}:
            raise ValueError("Input names do not match model")
        started = time.perf_counter()
        predictions = session.run(None, feeds)
        report["first_run_seconds"] = time.perf_counter() - started
        report["memory"]["after_first_run"] = resident_memory()
        del predictions
        for _ in range(warmup):
            session.run(None, feeds)
        report["memory"]["after_warmup"] = resident_memory()
        report["seconds"] = []
        for _ in range(repeats):
            started = time.perf_counter()
            predictions = session.run(None, feeds)
            report["seconds"].append(time.perf_counter() - started)
            # Do not retain the previous outputs while allocating the next batch.
            if any(not np.isfinite(value).all() for value in predictions):
                raise ValueError("Nonfinite model outputs")
            del predictions
        report["memory"]["after_measurement"] = resident_memory()
        report["median_seconds"] = statistics.median(report["seconds"])
        # Provenance inspection can deserialize large inline tensors. Do it only
        # after capturing the measurement peak, so it cannot inflate that result.
        import onnx

        report["versions"]["onnx"] = onnx.__version__
        report["model_files"] = model_files(model, onnx)
        report["inputs"] = {"path": str(inputs), "sha256": file_hash(inputs)}
        predictions = session.run(None, feeds)
        names = [node.name for node in session.get_outputs()]
        np.savez(output / "outputs.npz", **dict(zip(names, predictions, strict=True)))
        report["outputs"] = {"names": names, "sha256": file_hash(output / "outputs.npz")}
        report["status"] = "measured"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = ArgumentParser(description=__doc__)
    for name in ("model", "inputs", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=31)
    parser.add_argument("--optimization", choices=["all", "disable"], default="all")
    measure(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
