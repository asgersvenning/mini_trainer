"""Calibrate a fresh ONNX QDQ artifact from explicit, ordered input batches."""

import hashlib
import inspect
import io
import json
import math
import platform
from argparse import ArgumentParser
from collections import Counter
from pathlib import Path

import numpy as np

from .onnx_inference import file_hash, model_files


def calibration_manifest(path):
    manifest = json.loads(Path(path).read_text())
    if manifest.get("schema_version") != 1 or manifest.get("split") not in ("train", "calibration"):
        raise ValueError("Require schema_version=1 and a declared train/calibration split")
    if not isinstance(manifest.get("batch_input"), str) or not manifest["batch_input"]:
        raise ValueError("Declare the input whose leading dimension identifies samples: batch_input")
    if not isinstance(manifest.get("provenance"), dict) or not manifest["provenance"]:
        raise ValueError("Record dataset/preprocessing provenance")
    if not isinstance(manifest.get("batches"), list) or not manifest["batches"]:
        raise ValueError("Supply an ordered nonempty batches list")
    seen = set()
    for batch in manifest["batches"]:
        if not isinstance(batch.get("path"), str) or not batch["path"]:
            raise ValueError("Each batch needs an NPZ path")
        ids = batch.get("sample_ids")
        if not isinstance(ids, list) or not ids or any(not isinstance(x, str) or not x for x in ids):
            raise ValueError("Each batch needs nonempty string sample_ids")
        if len(set(ids)) != len(ids) or seen.intersection(ids):
            raise ValueError("Calibration sample IDs must be unique")
        seen.update(ids)
    return manifest


def load_batch(manifest_path, manifest, batch):
    path = (Path(manifest_path).parent / batch["path"]).resolve()
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if batch.get("sha256") is not None and batch["sha256"] != digest:
        raise ValueError(f"Calibration input hash mismatch: {path}")
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        feeds = {name: archive[name].copy(order="C") for name in archive.files}
    if not feeds or any(not np.isfinite(a).all() for a in feeds.values()):
        raise ValueError(f"Calibration arrays must be finite: {path}")
    name = manifest["batch_input"]
    if name not in feeds or feeds[name].ndim == 0 or feeds[name].shape[0] != len(batch["sample_ids"]):
        raise ValueError(f"sample_ids do not match the batch_input leading dimension: {path}")
    record = {
        "path": str(path),
        "sha256": digest,
        "sample_ids": batch["sample_ids"],
        "arrays": {name: {"shape": list(a.shape), "dtype": str(a.dtype)} for name, a in feeds.items()},
    }
    return feeds, record


def calibrate(
    model,
    manifest,
    output,
    method="percentile",
    percentile=99.9,
    activation_type="uint8",
    symmetric_activations=False,
    symmetric_calibration=False,
    float_bias=False,
    per_channel=True,
    op_types=("Conv", "Gemm", "MatMul"),
    threads=1,
):
    if method not in ("minmax", "percentile") or activation_type not in ("uint8", "int8"):
        raise ValueError("Choose minmax/percentile calibration and uint8/int8 activations")
    if not math.isfinite(percentile) or not 0 < percentile <= 100 or threads < 1 or not op_types:
        raise ValueError("Require percentile in (0,100], positive threads and selected operator types")
    metadata = calibration_manifest(manifest)
    try:
        import onnx
        import onnxruntime as ort
        from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static
        from onnxruntime.quantization.calibrate import MinMaxCalibrater, PercentileCalibrater, save_tensors_data
    except ImportError as error:
        raise ImportError("Use an explicitly prepared ONNX/ONNX Runtime quantization environment; this command installs nothing") from error
    if "calibration_cache_path" not in inspect.signature(quantize_static).parameters:
        raise RuntimeError("This command requires ONNX Runtime's calibration-cache API (tested with 1.29.0)")

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": 1,
        "status": "running",
        "runner_sha256": file_hash(__file__),
        "versions": {"onnx": onnx.__version__, "onnxruntime": ort.__version__, "numpy": np.__version__},
        "environment": {"platform": platform.platform(), "python": platform.python_version()},
        "calibration_manifest": {"path": str(manifest), "sha256": file_hash(manifest), "contents": metadata},
        "recipe": {
            "format": "QDQ",
            "method": method,
            "percentile": percentile if method == "percentile" else None,
            "histogram_symmetric": symmetric_calibration,
            "activation_type": activation_type,
            "activation_symmetric": symmetric_activations,
            "weight_type": "int8",
            "weight_symmetric": True,
            "per_channel_weights": per_channel,
            "quantize_bias": not float_bias,
            "op_types": list(op_types),
        },
        "execution": {
            "provider": "CPUExecutionProvider",
            "intra_op_threads": threads,
            "inter_op_threads": 1,
            "calibration_graph_optimization": "disabled",
            "collection_batch_limit": 1,
        },
        "batches": [],
        "scope": (
            "Calibration from declared training/calibration inputs and one calibration-batch execution; "
            "not held-out quality, integer-kernel placement or deployment acceptance."
        ),
    }
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    base = MinMaxCalibrater if method == "minmax" else PercentileCalibrater

    class Calibrator(base):
        def create_inference_session(self):
            self.infer_session = ort.InferenceSession(self.augmented_model_path, sess_options=options, providers=["CPUExecutionProvider"])
            self.infer_session.disable_fallback()

    class OneBatch(CalibrationDataReader):
        def __init__(self, feeds):
            self.feeds = feeds

        def get_next(self):
            feeds, self.feeds = self.feeds, None
            return feeds

    try:
        report["source_files"] = model_files(Path(model), onnx)
        source = onnx.load(model)
        report["source_ops"] = dict(Counter(node.op_type for node in source.graph.node))
        if not set(op_types).intersection(report["source_ops"]):
            raise ValueError("No selected operator types occur in the source graph")
        # ORT writes inferred-model sidecars beside its input. Isolate those writes.
        snapshot_dir = output / "source"
        snapshot_dir.mkdir()
        snapshot = snapshot_dir / "model.onnx"
        onnx.save_model(
            source, snapshot, save_as_external_data=True, all_tensors_to_one_file=True, location="weights.data", size_threshold=0
        )
        del source
        report["snapshot_files"] = model_files(snapshot, onnx)
        collector = Calibrator(
            snapshot,
            list(op_types),
            augmented_model_path=str(output / "calibration.onnx"),
            use_external_data_format=True,
            symmetric=symmetric_calibration,
            **({"percentile": percentile} if method == "percentile" else {}),
        )
        collector.augment_graph()
        collector.create_inference_session()
        names = {node.name for node in collector.infer_session.get_inputs()}
        for entry in metadata["batches"]:
            feeds, record = load_batch(manifest, metadata, entry)
            report["batches"].append(record)
            if set(feeds) != names:
                raise ValueError("Calibration input names do not match the source model")
            collector.collect_data(OneBatch(feeds))
            del feeds
        cache = output / "ranges.json"
        ranges = collector.compute_data()
        if not len(ranges.data):
            raise ValueError("No tensor ranges were calibrated")
        report["calibrated_tensors"] = len(ranges.data)
        save_tensors_data(ranges, cache)
        del collector, ranges
        report["ranges_sha256"] = file_hash(cache)
        quantized = output / "model.onnx"
        quantize_static(
            snapshot,
            quantized,
            None,
            quant_format=QuantFormat.QDQ,
            per_channel=per_channel,
            activation_type=QuantType.QUInt8 if activation_type == "uint8" else QuantType.QInt8,
            weight_type=QuantType.QInt8,
            op_types_to_quantize=list(op_types),
            calibration_cache_path=cache,
            use_external_data_format=True,
            calibrate_method=CalibrationMethod.MinMax if method == "minmax" else CalibrationMethod.Percentile,
            extra_options={"ActivationSymmetric": symmetric_activations, "WeightSymmetric": True, "QuantizeBias": not float_bias},
        )
        onnx.checker.check_model(str(quantized))
        graph = onnx.load(quantized, load_external_data=False)
        report["output_ops"] = dict(Counter(node.op_type for node in graph.graph.node))
        if not report["output_ops"].get("QuantizeLinear") or not report["output_ops"].get("DequantizeLinear"):
            raise ValueError("The output graph contains no complete QDQ quantization")
        del graph
        report["output_files"] = model_files(quantized, onnx)
        # A smoke execution of calibration data establishes loadability, not quality.
        feeds, record = load_batch(manifest, metadata, metadata["batches"][0])
        if record["sha256"] != report["batches"][0]["sha256"]:
            raise ValueError("First calibration batch changed before smoke execution")
        smoke_options = ort.SessionOptions()
        smoke_options.intra_op_num_threads = threads
        smoke_options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(quantized), sess_options=smoke_options, providers=["CPUExecutionProvider"])
        session.disable_fallback()
        arrays = session.run(None, feeds)
        if any(not np.isfinite(a).all() for a in arrays):
            raise ValueError("Quantized graph produced nonfinite calibration-smoke outputs")
        output_names = [node.name for node in session.get_outputs()]
        np.savez(output / "calibration-smoke.npz", **dict(zip(output_names, arrays, strict=True)))
        report["smoke"] = {
            "batch_index": 0,
            "graph_optimization": "all",
            "outputs": output_names,
            "sha256": file_hash(output / "calibration-smoke.npz"),
        }
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
    parser.add_argument("--manifest", type=Path, required=True, help="Ordered calibration NPZ batches, sample IDs and provenance")
    parser.add_argument("--output", type=Path, required=True, help="New artifact directory")
    parser.add_argument("--method", choices=["minmax", "percentile"], default="percentile")
    parser.add_argument("--percentile", type=float, default=99.9)
    parser.add_argument("--activation-type", choices=["uint8", "int8"], default="uint8")
    parser.add_argument("--symmetric-activations", action="store_true")
    parser.add_argument("--symmetric-calibration", action="store_true", help="Collect symmetric calibration ranges/histograms")
    parser.add_argument("--float-bias", action="store_true", help="Keep biases floating instead of quantizing them to INT32")
    parser.add_argument("--per-tensor-weights", dest="per_channel", action="store_false")
    parser.add_argument("--op-type", dest="op_types", action="append", help="Operator type to quantize; repeatable")
    parser.add_argument("--threads", type=int, default=1)
    args = vars(parser.parse_args())
    if args["op_types"] is None:
        args["op_types"] = ["Conv", "Gemm", "MatMul"]
    calibrate(**args)


if __name__ == "__main__":
    main()
