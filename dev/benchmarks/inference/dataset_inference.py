"""Collect held-out ONNX/TensorRT predictions for the paired quality evaluator."""

import csv
import hashlib
import json
import platform
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from .onnx_calibration import load_batch
from .onnx_inference import file_hash, model_files
from .quality_compare import COLUMNS, validate_dataset


def inference_manifest(path):
    payload = Path(path).read_bytes()
    manifest = json.loads(payload)
    validate_dataset(manifest)
    if not isinstance(manifest.get("batch_input"), str) or not manifest["batch_input"]:
        raise ValueError("Declare batch_input for sample identity")
    outputs = []
    for level in manifest["levels"]:
        if (
            not isinstance(level.get("output"), str)
            or not level["output"]
            or level.get("score_semantics") not in ("logits", "probabilities")
        ):
            raise ValueError("Each level needs an output name and logits/probabilities score_semantics")
        outputs.append(level["output"])
    if len(set(outputs)) != len(outputs):
        raise ValueError("Output bindings must be unique")
    batches = manifest.get("batches")
    if not isinstance(batches, list) or not batches:
        raise ValueError("Supply ordered input batches")
    ids = []
    for batch in batches:
        if not isinstance(batch.get("path"), str) or not batch["path"]:
            raise ValueError("Each batch needs an NPZ path")
        if not isinstance(batch.get("sample_ids"), list) or not batch["sample_ids"]:
            raise ValueError("Each batch needs sample_ids as canonical integer strings")
        ids.extend(batch["sample_ids"])
    if (
        any(not isinstance(i, str) for i in ids)
        or len(set(ids)) != len(ids)
        or set(ids) != {str(s["instance_id"]) for s in manifest["samples"]}
    ):
        raise ValueError("Batches must cover every declared sample exactly once using canonical string IDs")
    return manifest, hashlib.sha256(payload).hexdigest()


def predictions(values, level, count):
    values = np.asarray(values)
    if values.shape != (count, len(level["classes"])) or values.dtype.kind != "f" or not np.isfinite(values).all():
        raise ValueError(f"Expected finite floating [batch,classes] scores for {level['name']}")
    indices = values.argmax(axis=1)
    if level["score_semantics"] == "probabilities":
        if (values < 0).any() or (values > 1).any() or not np.allclose(values.sum(axis=1, dtype=np.float64), 1, rtol=1e-5, atol=1e-5):
            raise ValueError("Declared probabilities must lie in [0,1] and sum to one")
        confidence = values[np.arange(count), indices]
    else:
        shifted = values.astype(np.float64) - values.max(axis=1, keepdims=True)
        confidence = 1 / np.exp(shifted).sum(axis=1)
    return indices, confidence


class OnnxPredictor:
    def __init__(self, model, report, provider, provider_options, threads, optimization):
        import onnx
        import onnxruntime as ort

        if provider not in ort.get_available_providers():
            raise ValueError(f"Requested provider unavailable: {provider}")
        report["model_files"] = model_files(Path(model), onnx)
        options = ort.SessionOptions()
        options.intra_op_num_threads, options.inter_op_num_threads = threads, 1
        options.graph_optimization_level = getattr(
            ort.GraphOptimizationLevel, f"ORT_{'DISABLE_ALL' if optimization == 'disable' else 'ENABLE_ALL'}"
        )
        self.session = ort.InferenceSession(str(model), sess_options=options, providers=[provider], provider_options=[provider_options])
        self.session.disable_fallback()
        if provider not in self.session.get_providers():
            raise RuntimeError("Requested provider was not activated")
        self.names = [node.name for node in self.session.get_outputs()]
        report["runtime"] = {
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
            "providers": self.session.get_providers(),
            "provider_options": self.session.get_provider_options(),
        }

    def __call__(self, feeds):
        return dict(zip(self.names, self.session.run(None, feeds), strict=True))


class TensorRTPredictor:
    def __init__(self, model, report, device):
        import tensorrt as trt
        import torch

        self.torch, self.trt, self.device = torch, trt, device
        self.logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self.logger, "")
        self.runtime = trt.Runtime(self.logger)
        payload = Path(model).read_bytes()
        report["model_files"] = [{"path": str(Path(model).resolve()), "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}]
        with torch.cuda.device(device):
            self.engine = self.runtime.deserialize_cuda_engine(payload)
            if self.engine is None:
                raise RuntimeError("Could not deserialize TensorRT engine")
            self.context = self.engine.create_execution_context()
            if self.context is None:
                raise RuntimeError("Could not create TensorRT execution context")
            self.stream = torch.cuda.Stream(device=device)
            report["runtime"] = {
                "tensorrt": trt.__version__,
                "torch": torch.__version__,
                "gpu": torch.cuda.get_device_name(device),
                "device": device,
                "profile": 0,
            }

    def __call__(self, feeds):
        from .tensorrt_pair import buffers_for

        torch = self.torch
        with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
            buffers, inputs = buffers_for(self.engine, self.context, feeds, torch, self.trt, self.device, False)
            for name in inputs:
                gpu, host = buffers[name]
                gpu.copy_(host)
            if not self.context.execute_async_v3(self.stream.cuda_stream):
                raise RuntimeError("TensorRT dataset execution failed")
            for name, (gpu, host) in buffers.items():
                if name not in inputs:
                    host.copy_(gpu)
            self.stream.synchronize()
            return {name: host.numpy() for name, (_, host) in buffers.items() if name not in inputs}


def pair_bundle(baseline, candidate):
    old = json.loads(Path(baseline).read_text())
    for key in ("schema_version", "split", "levels", "samples"):
        if old[key] != candidate[key]:
            raise ValueError(f"Baseline bundle differs in held-out {key}")
    return {
        **{key: candidate[key] for key in ("schema_version", "split", "levels", "samples", "provenance")},
        "baseline": old["artifact"],
        "candidate": candidate["artifact"],
    }


def collect(
    model,
    manifest,
    output,
    backend="onnx",
    provider="CPUExecutionProvider",
    provider_options=None,
    threads=1,
    optimization="all",
    device=0,
    save_scores=False,
    baseline_bundle=None,
):
    if backend not in ("onnx", "tensorrt") or threads < 1 or device < 0 or optimization not in ("all", "disable"):
        raise ValueError("Invalid backend, thread count, device or graph optimization")
    metadata, digest = inference_manifest(manifest)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": 1,
        "status": "running",
        "runner_sha256": file_hash(__file__),
        "environment": {"platform": platform.platform(), "python": platform.python_version(), "numpy": np.__version__},
        "manifest": {"path": str(manifest), "sha256": digest, "contents": metadata},
        "settings": {
            "backend": backend,
            "provider": provider if backend == "onnx" else None,
            "threads": threads if backend == "onnx" else None,
            "optimization": optimization if backend == "onnx" else None,
            "save_scores": save_scores,
        },
        "batches": [],
        "scope": (
            "Held-out prediction collection from explicit preprocessed inputs; "
            "not speed, integer placement, score parity or deployment acceptance."
        ),
    }
    try:
        predictor = (
            OnnxPredictor(model, report, provider, provider_options or {}, threads, optimization)
            if backend == "onnx"
            else TensorRTPredictor(model, report, device)
        )
        samples = {str(s["instance_id"]): s for s in metadata["samples"]}
        path = output / "predictions.csv"
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=COLUMNS)
            writer.writeheader()
            for index, batch in enumerate(metadata["batches"]):
                feeds, record = load_batch(manifest, metadata, batch)
                arrays = predictor(feeds)
                rows = []
                for level, spec in enumerate(metadata["levels"]):
                    if spec["output"] not in arrays:
                        raise ValueError(f"Missing mapped output: {spec['output']}")
                    predicted, confidence = predictions(arrays[spec["output"]], spec, len(batch["sample_ids"]))
                    for identifier, pred, conf in zip(batch["sample_ids"], predicted, confidence, strict=True):
                        sample = samples[identifier]
                        rows.append(
                            dict(
                                instance_id=sample["instance_id"],
                                filename=sample["filename"],
                                level=level,
                                label=sample["labels"][level],
                                prediction=spec["classes"][int(pred)],
                                confidence=float(conf),
                                threshold=0,
                            )
                        )
                if save_scores:
                    score_path = output / f"scores-{index:05d}.npz"
                    np.savez(score_path, **arrays)
                    record["scores"] = {"path": score_path.name, "sha256": file_hash(score_path)}
                writer.writerows(rows)
                report["batches"].append(record)
                del feeds, arrays
        artifact = {
            "path": str(path.resolve()),
            "sha256": file_hash(path),
            "classes": [level["classes"] for level in metadata["levels"]],
            "provenance": {
                "model_files": report["model_files"],
                "inference_manifest_sha256": digest,
                "runner_sha256": report["runner_sha256"],
                "runtime": report["runtime"],
                "report": str((output / "report.json").resolve()),
            },
        }
        bundle = {
            **{key: metadata[key] for key in ("schema_version", "split", "samples", "provenance")},
            "levels": [{key: spec[key] for key in ("name", "classes")} for spec in metadata["levels"]],
            "artifact": artifact,
        }
        (output / "evaluation.json").write_text(json.dumps(bundle, indent=2) + "\n")
        if baseline_bundle is not None:
            (output / "comparison.json").write_text(json.dumps(pair_bundle(baseline_bundle, bundle), indent=2) + "\n")
        report.update(status="inferred", predictions=artifact)
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=["onnx", "tensorrt"], default="onnx")
    parser.add_argument("--provider", default="CPUExecutionProvider")
    parser.add_argument("--provider-options", type=json.loads)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--optimization", choices=["all", "disable"], default="all")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--save-scores", action="store_true")
    parser.add_argument("--baseline-bundle", type=Path, help="Create comparison.json against a previous inference evaluation.json")
    collect(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
