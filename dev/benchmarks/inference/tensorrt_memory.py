"""Collect single-engine TensorRT memory snapshots in a fresh Linux process."""

import hashlib
import io
import json
import platform
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from .onnx_cpu_memory import resident_memory
from .onnx_inference import file_hash
from .tensorrt_pair import buffers_for


def memory_snapshot(torch, device):
    """Keep device-wide observations separate from process and allocator counters."""
    torch.cuda.synchronize(device)
    free, total = torch.cuda.mem_get_info(device)
    return {
        "device_used_bytes": total - free,
        "device_total_bytes": total,
        "torch_allocated_bytes": torch.cuda.memory_allocated(device),
        "torch_reserved_bytes": torch.cuda.memory_reserved(device),
        "host": resident_memory(),
    }


def measure(engine, inputs, output, runs=20, threads=1, device=0, pinned=False):
    """Call the CLI separately for every engine/trial; no environment installation."""
    if runs < 1 or threads < 1 or device < 0:
        raise ValueError("Require positive runs/threads and a nonnegative device")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": 1,
        "status": "running",
        "runner_sha256": file_hash(__file__),
        "environment": {"platform": platform.platform(), "python": platform.python_version()},
        "settings": {"runs": runs, "threads": threads, "device": device, "pinned_host_io": pinned, "profile": 0},
        "memory": {},
        "messages": [],
        "scope": (
            "Run one engine per fresh Linux process. Device used/free is device-wide, not per-process or a transient peak; "
            "other activity and WSL accounting can affect attribution. PyTorch counters exclude TensorRT-owned allocations. "
            "Host RSS/PSS includes interpreter, imports, inputs, engine and IO; host peak is approximate since exec. "
            "No latency or quality acceptance claim. Snapshots precede output serialization."
        ),
    }
    try:
        report["memory"]["before_runtime_import"] = {"host": resident_memory()}
        try:
            import tensorrt as trt
            import torch
        except ImportError as error:
            raise ImportError("Use an explicitly prepared TensorRT and CUDA PyTorch environment; this command installs nothing") from error
        torch.set_num_threads(threads)
        report["versions"] = {"tensorrt": trt.__version__, "torch": torch.__version__, "numpy": np.__version__}
        payload = Path(inputs).read_bytes()
        report["inputs"] = {"path": str(inputs), "sha256": hashlib.sha256(payload).hexdigest()}
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            feeds = {name: archive[name].copy(order="C") for name in archive.files}
        del payload
        if not feeds or any(not np.isfinite(a).all() for a in feeds.values()):
            raise ValueError("Supply finite named input arrays")

        class Logger(trt.ILogger):
            def log(self, severity, message):
                if severity <= trt.ILogger.WARNING:
                    report["messages"].append({"severity": str(severity), "message": message})

        with torch.cuda.device(device):
            stream = torch.cuda.Stream(device=device)
            report["environment"].update(
                gpu=torch.cuda.get_device_name(device), compute_capability=list(torch.cuda.get_device_capability(device))
            )
            report["memory"]["cuda_initialized"] = memory_snapshot(torch, device)
            logger = Logger()
            trt.init_libnvinfer_plugins(logger, "")
            runtime = trt.Runtime(logger)
            serialized = Path(engine).read_bytes()
            report["engine"] = {"path": str(engine), "sha256": hashlib.sha256(serialized).hexdigest(), "bytes": len(serialized)}
            model = runtime.deserialize_cuda_engine(serialized)
            del serialized
            if model is None:
                raise RuntimeError("Could not deserialize engine")
            report["memory"]["engine_loaded"] = memory_snapshot(torch, device)
            context = model.create_execution_context()
            if context is None:
                raise RuntimeError("Could not create execution context")
            with torch.cuda.stream(stream):
                buffers, input_names = buffers_for(model, context, feeds, torch, trt, device, pinned)
            report["io"] = {
                name: {"shape": list(host.shape), "dtype": str(host.numpy().dtype), "input": name in input_names}
                for name, (_, host) in buffers.items()
            }
            if all(item["input"] for item in report["io"].values()):
                raise ValueError("Require at least one engine output")
            report["context_memory_bytes"] = model.device_memory_size_v2
            report["memory"]["context_and_io"] = memory_snapshot(torch, device)
            for _ in range(runs):
                with torch.cuda.stream(stream):
                    for name in input_names:
                        gpu, host = buffers[name]
                        gpu.copy_(host, non_blocking=pinned)
                    if not context.execute_async_v3(stream.cuda_stream):
                        raise RuntimeError("Engine execution failed")
                    for name, (gpu, host) in buffers.items():
                        if name not in input_names:
                            host.copy_(gpu, non_blocking=pinned)
                stream.synchronize()
                if any(not np.isfinite(host.numpy()).all() for name, (_, host) in buffers.items() if name not in input_names):
                    raise ValueError("Nonfinite engine outputs")
            report["memory"]["warm"] = memory_snapshot(torch, device)
            arrays = {name: host.numpy() for name, (_, host) in buffers.items() if name not in input_names}
            np.savez(output / "outputs.npz", **arrays)
            report["outputs_sha256"] = file_hash(output / "outputs.npz")
            report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = ArgumentParser(description=__doc__)
    for name in ("engine", "inputs", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--pinned", action="store_true")
    measure(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
