"""Validate CUDA graph optimizations once before admitting a session for inference."""

import time
import warnings

import numpy as np

from .preprocessing import RECIPE


def create_session(ort, path, providers, threads, *, cuda, embeddings):
    started = time.perf_counter()
    attempts = []
    outputs = ["output_0", "embedding"] if embeddings else ["output_0"]
    profiles = [("optimized", ort.GraphOptimizationLevel.ORT_ENABLE_ALL)]
    if cuda:
        profiles.append(("unoptimized", ort.GraphOptimizationLevel.ORT_DISABLE_ALL))
    for name, level in profiles:
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.enable_profiling = False
        options.graph_optimization_level = level
        session = None
        probe_seconds = 0.0
        try:
            session = ort.InferenceSession(str(path), sess_options=options, providers=providers)
            session.disable_fallback()
            if cuda and session.get_providers()[0] != "CUDAExecutionProvider":
                raise RuntimeError("Requested ONNX CUDA provider failed to initialize; refusing CPU-only fallback")
            if cuda:
                # CPU outputs make completion synchronous. No image decoding or dataset access.
                probe_started = time.perf_counter()
                session.run(outputs, {"images": np.zeros((1, 3, RECIPE["crop_size"], RECIPE["crop_size"]), dtype=np.float32)})
                probe_seconds = time.perf_counter() - probe_started
        except Exception as error:
            session = None
            compatibility_error = any(code in str(error) for code in ("cudaErrorNoKernelImageForDevice", "cudaErrorInvalidDeviceFunction"))
            if not cuda or not compatibility_error:
                raise
            attempts.append({"profile": name, "error": str(error)})
            if name == "unoptimized":
                raise RuntimeError(
                    "ONNX CUDA baseline compatibility check failed with graph optimizations disabled. "
                    "This runtime/device cannot execute the baseline model; use a compatible CUDA runtime build. "
                    "No CPU fallback was performed."
                ) from error
            continue
        if name == "unoptimized":
            warnings.warn(
                "ONNX CUDA optimized graph failed its kernel compatibility probe; using a validated session with "
                "graph optimizations disabled. GPU execution is retained; throughput may be lower.",
                RuntimeWarning,
                stacklevel=2,
            )
        return session, {
            "profile": name,
            "probe_batch_size": 1 if cuda else None,
            "probe_seconds": probe_seconds,
            "initialization_seconds": time.perf_counter() - started,
            "failed_attempts": attempts,
        }
