"""Measure paired TensorRT engine latency with explicit inputs and retained trials."""

import hashlib
import io
import json
import math
import platform
import statistics
import time
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from .onnx_inference import file_hash


def paired_trials(execute, warmup, repeats, reverse=False):
    """Alternate adjacent measurements; keep each ratio paired within its trial."""
    if warmup < 0 or repeats < 1:
        raise ValueError("Require nonnegative warmup and positive repeats")
    for index in range(warmup + repeats):
        order = ["baseline", "candidate"]
        if bool(index % 2) ^ reverse:
            order.reverse()
        seconds = {name: execute(name) for name in order}
        if any(not math.isfinite(value) or value <= 0 for value in seconds.values()):
            raise ValueError("Measured durations must be finite and positive")
        if index >= warmup:
            yield {"index": index - warmup, "order": order, "seconds": seconds}


def summarize(trials):
    ratios = [trial["seconds"]["candidate"] / trial["seconds"]["baseline"] for trial in trials]
    return {
        "median_seconds": {name: statistics.median(trial["seconds"][name] for trial in trials) for name in ("baseline", "candidate")},
        "candidate_over_baseline_ratios": ratios,
        "median_paired_ratio": statistics.median(ratios),
    }


def buffers_for(engine, context, feeds, torch, trt, device, pinned):
    names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    inputs = {name for name in names if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT}
    if inputs != set(feeds):
        raise ValueError("Input names do not match engine")
    for name in names:
        if engine.get_tensor_format(name) != trt.TensorFormat.LINEAR or engine.get_tensor_location(name) != trt.TensorLocation.DEVICE:
            raise ValueError(f"Require linear device IO: {name}")
    for name, value in feeds.items():
        if engine.is_shape_inference_io(name):
            raise ValueError(f"Shape-tensor inputs are not supported by this probe: {name}")
        if np.dtype(trt.nptype(engine.get_tensor_dtype(name))) != value.dtype:
            raise ValueError(f"Input dtype does not match engine: {name}")
        if not context.set_input_shape(name, value.shape):
            raise ValueError(f"Input shape is outside engine profile 0: {name}")
    unresolved = context.infer_shapes()
    if unresolved:
        raise ValueError(f"Unresolved engine shapes: {unresolved}")
    buffers = {}
    for name in names:
        shape = tuple(context.get_tensor_shape(name))
        if any(n <= 0 for n in shape):
            raise ValueError(f"Unresolved or empty tensor: {name}")
        dtype = torch.from_numpy(np.empty((), dtype=trt.nptype(engine.get_tensor_dtype(name)))).dtype
        host = torch.empty(shape, dtype=dtype, pin_memory=pinned)
        if name in inputs:
            host.copy_(torch.from_numpy(feeds[name]))
        gpu = torch.empty(shape, dtype=dtype, device=f"cuda:{device}")
        if not context.set_tensor_address(name, gpu.data_ptr()):
            raise RuntimeError(f"TensorRT rejected IO address: {name}")
        buffers[name] = (gpu, host)
    return buffers, tuple(sorted(inputs))


def benchmark(baseline, candidate, inputs, output, warmup=10, repeats=31, reverse=False, device=0, pinned=False):
    if warmup < 0 or repeats < 1 or device < 0:
        raise ValueError("Require nonnegative warmup/device and positive repeats")
    payload = Path(inputs).read_bytes()
    input_hash = hashlib.sha256(payload).hexdigest()
    with np.load(io.BytesIO(payload), allow_pickle=False) as data:
        feeds = {name: data[name].copy(order="C") for name in data.files}
    del payload
    if not feeds or any(not np.isfinite(a).all() for a in feeds.values()):
        raise ValueError("Supply finite named input arrays")
    try:
        import tensorrt as trt
        import torch
    except ImportError as error:
        raise ImportError("Use an explicitly prepared TensorRT and CUDA PyTorch environment; this command installs nothing") from error

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": 1,
        "status": "running",
        "runner_sha256": file_hash(__file__),
        "versions": {"tensorrt": trt.__version__, "torch": torch.__version__, "numpy": np.__version__},
        "environment": {"platform": platform.platform(), "python": platform.python_version()},
        "settings": {"warmup": warmup, "repeats": repeats, "reverse": reverse, "device": device, "pinned_host_io": pinned, "profile": 0},
        "inputs": {"path": str(inputs), "sha256": input_hash},
        "models": {},
        "messages": [],
        "scope": (
            "Synchronized host latency: preallocated host-to-device inputs, engine execution and device-to-host outputs. "
            "Excludes decoding/preprocessing, allocation, deserialization and shape changes. Both engines/contexts coexist. "
            "No CUDA-event timing or CUDA graph capture. Not a quality, integer-placement or total-memory acceptance test."
        ),
    }

    class Logger(trt.ILogger):
        def log(self, severity, message):
            if severity <= trt.ILogger.WARNING:
                report["messages"].append({"severity": str(severity), "message": message})

    try:
        with torch.cuda.device(device):
            report["environment"].update(
                gpu=torch.cuda.get_device_name(device), compute_capability=list(torch.cuda.get_device_capability(device))
            )
            report["environment"].update(torch_threads=torch.get_num_threads(), torch_interop_threads=torch.get_num_interop_threads())
            logger = Logger()
            trt.init_libnvinfer_plugins(logger, "")
            runtime = trt.Runtime(logger)
            stream = torch.cuda.Stream(device=device)
            models = {}
            for name, path in (("baseline", baseline), ("candidate", candidate)):
                before = time.perf_counter()
                serialized = Path(path).read_bytes()
                read_seconds = time.perf_counter() - before
                info = {
                    "path": str(path),
                    "sha256": hashlib.sha256(serialized).hexdigest(),
                    "bytes": len(serialized),
                    "read_seconds": read_seconds,
                }
                report["models"][name] = info
                torch.cuda.synchronize(device)
                before = time.perf_counter()
                engine = runtime.deserialize_cuda_engine(serialized)
                if engine is None:
                    raise RuntimeError(f"Could not deserialize {name} engine")
                context = engine.create_execution_context()
                if context is None:
                    raise RuntimeError(f"Could not create {name} execution context")
                torch.cuda.synchronize(device)
                info["deserialize_and_context_seconds"] = time.perf_counter() - before
                del serialized
                with torch.cuda.stream(stream):
                    buffers, input_names = buffers_for(engine, context, feeds, torch, trt, device, pinned)
                info["context_memory_bytes"] = engine.device_memory_size_v2
                info["io"] = {
                    key: {"shape": list(host.shape), "dtype": str(host.numpy().dtype), "input": key in input_names}
                    for key, (_, host) in buffers.items()
                }
                models[name] = engine, context, buffers, input_names
            # Outputs can use different precisions but must describe the same named shapes.
            contracts = [
                {key: value["shape"] for key, value in info["io"].items() if not value["input"]} for info in report["models"].values()
            ]
            if not contracts[0] or contracts[0] != contracts[1]:
                raise ValueError("Engine output names/shapes must match")
            stream.synchronize()

            def execute(name):
                _, context, buffers, input_names = models[name]
                with torch.cuda.stream(stream):
                    before = time.perf_counter()
                    for key in input_names:
                        gpu, host = buffers[key]
                        gpu.copy_(host, non_blocking=pinned)
                    if not context.execute_async_v3(stream.cuda_stream):
                        raise RuntimeError(f"Execution failed: {name}")
                    for key, (gpu, host) in buffers.items():
                        if key not in input_names:
                            host.copy_(gpu, non_blocking=pinned)
                    stream.synchronize()
                    return time.perf_counter() - before

            report["trials"] = []
            report["trials"].extend(paired_trials(execute, warmup, repeats, reverse))
            report["summary"] = summarize(report["trials"])
            for name, (_, _, buffers, input_names) in models.items():
                arrays = {key: host.numpy() for key, (_, host) in buffers.items() if key not in input_names}
                np.savez(output / f"{name}-outputs.npz", **arrays)
                report["models"][name]["outputs_sha256"] = file_hash(output / f"{name}-outputs.npz")
                if any(not np.isfinite(a).all() for a in arrays.values()):
                    raise ValueError(f"Nonfinite final outputs: {name}")
            report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True, help="Named preprocessed NPZ inputs with exact shapes and dtypes")
    parser.add_argument("--output", type=Path, required=True, help="New report/output directory")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=31)
    parser.add_argument("--reverse", action="store_true", help="Reverse the alternating execution order")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--pinned", action="store_true", help="Use preallocated pinned host IO and nonblocking copies")
    benchmark(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
