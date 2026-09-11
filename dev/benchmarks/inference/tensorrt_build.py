"""Build, inspect and smoke-test an ONNX TensorRT engine in an explicit environment."""

import json
import math
import platform
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from .onnx_inference import file_hash, model_files


def input_profiles(feeds, profiles=None):
    """Validate explicit ranges; absent ranges use the supplied fixed input shapes."""
    if profiles is None:
        profiles = {name: {key: list(value.shape) for key in ("min", "opt", "max")} for name, value in feeds.items()}
    if set(profiles) != set(feeds):
        raise ValueError("Profile names must match the named input arrays")
    result = {}
    for name, value in feeds.items():
        ranges = profiles[name]
        if set(ranges) != {"min", "opt", "max"}:
            raise ValueError(f"Profile {name} needs min, opt and max shapes")
        for shape in ranges.values():
            if len(shape) != value.ndim or any(type(n) is not int or n < 1 for n in shape):
                raise ValueError(f"Invalid dimensions in profile {name}")
        for low, opt, high, sample in zip(ranges["min"], ranges["opt"], ranges["max"], value.shape, strict=True):
            if not low <= opt <= high or not low <= sample <= high:
                raise ValueError(f"Unordered ranges or sample outside profile {name}")
        result[name] = {key: list(shape) for key, shape in ranges.items()}
    return result


def compare_outputs(actual, expected, rtol, atol):
    if set(actual) != set(expected):
        raise ValueError("Reference output names do not match engine outputs")
    result = {}
    for name, value in actual.items():
        reference = expected[name]
        if value.shape != reference.shape or value.dtype != reference.dtype:
            raise ValueError(f"Reference shape/dtype does not match output {name}")
        if not np.isfinite(reference).all():
            raise ValueError(f"Nonfinite reference output {name}")
        close = np.isclose(value, reference, rtol=rtol, atol=atol) if value.dtype.kind == "f" else value == reference
        result[name] = {"passed": bool(close.all()), "elements_outside_tolerance": int((~close).sum())}
    return result


def build(
    model,
    inputs,
    output,
    profiles=None,
    fp16=False,
    tf32=False,
    workspace_mib=1024,
    optimization=3,
    device=0,
    reference=None,
    rtol=1e-4,
    atol=1e-5,
):
    if workspace_mib < 1 or not 0 <= optimization <= 5 or device < 0:
        raise ValueError("Require positive workspace, optimization 0–5 and nonnegative device index")
    if any(not math.isfinite(x) or x < 0 for x in (rtol, atol)):
        raise ValueError("Parity tolerances must be finite and nonnegative")
    with np.load(inputs, allow_pickle=False) as data:
        feeds = {name: data[name].copy(order="C") for name in data.files}
    if not feeds or any(not np.isfinite(a).all() for a in feeds.values()):
        raise ValueError("Supply finite named input arrays")
    profiles = input_profiles(feeds, profiles)
    try:
        import onnx
        import tensorrt as trt
        import torch
    except ImportError as error:
        raise ImportError(
            "Use an explicitly prepared compatible TensorRT, ONNX and CUDA PyTorch environment; no packages are installed by this command"
        ) from error

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": 1,
        "status": "running",
        "runner_sha256": file_hash(__file__),
        "versions": {"tensorrt": trt.__version__, "torch": torch.__version__, "onnx": onnx.__version__, "numpy": np.__version__},
        "environment": {"platform": platform.platform(), "python": platform.python_version()},
        "settings": {
            "profiles": profiles,
            "fp16_allowed": fp16,
            "tf32_allowed": tf32,
            "workspace_bytes": workspace_mib * 1024**2,
            "builder_optimization": optimization,
            "device": device,
        },
        "inputs": {
            "path": str(inputs),
            "sha256": file_hash(inputs),
            "arrays": {name: {"shape": list(a.shape), "dtype": str(a.dtype)} for name, a in feeds.items()},
        },
        "reference": None,
        "messages": [],
        "scope": (
            "Engine build, detailed layer inspection and one input execution; not a quality or performance certification. "
            "Workspace/context bytes are not total GPU memory."
        ),
    }

    class Logger(trt.ILogger):
        def log(self, severity, message):
            if severity <= trt.ILogger.WARNING:
                report["messages"].append({"severity": str(severity), "message": message})

    try:
        report["model_files"] = model_files(Path(model), onnx)
        with torch.cuda.device(device):
            report["environment"]["gpu"] = torch.cuda.get_device_name(device)
            report["environment"]["compute_capability"] = list(torch.cuda.get_device_capability(device))
            logger = Logger()
            trt.init_libnvinfer_plugins(logger, "")
            builder = trt.Builder(logger)
            network = builder.create_network(0)
            parser = trt.OnnxParser(network, logger)
            if not parser.parse_from_file(str(model)):
                report["parser_errors"] = [str(parser.get_error(i)) for i in range(parser.num_errors)]
                raise RuntimeError("TensorRT could not parse the model; see parser_errors in report.json")
            tensors = {network.get_input(i).name: network.get_input(i) for i in range(network.num_inputs)}
            if set(tensors) != set(feeds):
                raise ValueError("Input names do not match the parsed network")
            profile = builder.create_optimization_profile()
            for name, tensor in tensors.items():
                if tensor.is_shape_tensor:
                    raise ValueError(f"Shape-tensor input profiles are not supported by this probe: {name}")
                if np.dtype(trt.nptype(tensor.dtype)) != feeds[name].dtype:
                    raise ValueError(f"Input dtype does not match network: {name}")
                for shape in profiles[name].values():
                    if len(shape) != len(tensor.shape) or any(n >= 0 and n != s for n, s in zip(tensor.shape, shape, strict=True)):
                        raise ValueError(f"Profile changes a fixed network dimension: {name}")
                profile.set_shape(name, profiles[name]["min"], profiles[name]["opt"], profiles[name]["max"])
            config = builder.create_builder_config()
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mib * 1024**2)
            config.builder_optimization_level = optimization
            config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
            config.clear_flag(trt.BuilderFlag.TF32)
            if tf32:
                config.set_flag(trt.BuilderFlag.TF32)
            if fp16:
                config.set_flag(trt.BuilderFlag.FP16)
            if config.add_optimization_profile(profile) < 0:
                raise ValueError("TensorRT rejected the optimization profile")
            serialized = builder.build_serialized_network(network, config)
            if serialized is None:
                raise RuntimeError("TensorRT engine construction failed; see messages in report.json")
            engine_path = output / "model.engine"
            engine_path.write_bytes(serialized)
            runtime = trt.Runtime(logger)
            engine = runtime.deserialize_cuda_engine(serialized)
            if engine is None:
                raise RuntimeError("TensorRT could not deserialize the constructed engine")
            layers_path = output / "layers.json"
            layers_path.write_text(engine.create_engine_inspector().get_engine_information(trt.LayerInformationFormat.JSON))
            report["engine"] = {
                "sha256": file_hash(engine_path),
                "bytes": engine_path.stat().st_size,
                "context_memory_bytes": engine.device_memory_size_v2,
                "layers_sha256": file_hash(layers_path),
                "num_layers": engine.num_layers,
            }
            context = engine.create_execution_context()
            stream = torch.cuda.Stream(device=device)
            buffers, outputs = {}, {}
            with torch.cuda.stream(stream):
                for name, value in feeds.items():
                    buffers[name] = torch.from_numpy(value).to(device=f"cuda:{device}")
                    if not context.set_input_shape(name, value.shape):
                        raise ValueError(f"TensorRT rejected sample shape for {name}")
                for index in range(engine.num_io_tensors):
                    name = engine.get_tensor_name(index)
                    if (
                        engine.get_tensor_format(name) != trt.TensorFormat.LINEAR
                        or engine.get_tensor_location(name) != trt.TensorLocation.DEVICE
                    ):
                        raise ValueError(f"Probe requires linear device IO tensors: {name}")
                    if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                        shape = tuple(context.get_tensor_shape(name))
                        if any(n <= 0 for n in shape):
                            raise ValueError(f"Unresolved or empty output shape for {name}: {shape}")
                        dtype = torch.from_numpy(np.empty((), dtype=trt.nptype(engine.get_tensor_dtype(name)))).dtype
                        buffers[name] = torch.empty(shape, device=f"cuda:{device}", dtype=dtype)
                        outputs[name] = buffers[name]
                    if not context.set_tensor_address(name, buffers[name].data_ptr()):
                        raise RuntimeError(f"TensorRT rejected IO address for {name}")
                if not context.execute_async_v3(stream.cuda_stream):
                    raise RuntimeError("TensorRT execution failed")
                stream.synchronize()
                actual = {name: value.cpu().numpy() for name, value in outputs.items()}
            if any(not np.isfinite(a).all() for a in actual.values()):
                raise ValueError("Engine produced nonfinite outputs")
            np.savez(output / "outputs.npz", **actual)
            report["outputs"] = {name: {"shape": list(a.shape), "dtype": str(a.dtype)} for name, a in actual.items()}
            if reference is not None:
                report["reference"] = {"path": str(reference), "sha256": file_hash(reference), "rtol": rtol, "atol": atol}
                with np.load(reference, allow_pickle=False) as expected:
                    report["reference"]["comparison"] = compare_outputs(actual, expected, rtol, atol)
                if not all(r["passed"] for r in report["reference"]["comparison"].values()):
                    raise ValueError("Engine outputs failed reference parity")
            report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True, help="Named preprocessed input arrays in NPZ format")
    parser.add_argument("--output", type=Path, required=True, help="New directory for engine, inspection, outputs and report")
    parser.add_argument("--profiles", type=Path, help="JSON object mapping each input name to min/opt/max shape lists")
    parser.add_argument("--fp16", action="store_true", help="Allow FP16 tactics; QDQ quantization is defined by the graph")
    parser.add_argument("--tf32", action="store_true")
    parser.add_argument("--workspace-mib", type=int, default=1024)
    parser.add_argument("--optimization", type=int, choices=range(6), default=3)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--reference", type=Path, help="Optional NPZ of expected outputs with exact names/shapes/dtypes")
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-5)
    args = vars(parser.parse_args())
    if args["profiles"] is not None:
        args["profiles"] = json.loads(args["profiles"].read_text())
    build(**args)


if __name__ == "__main__":
    main()
