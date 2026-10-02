"""UCloud INT8 pilot: preserve splits, reuse quantizers, retain independent backend reports."""

import argparse
import csv
import hashlib
import heapq
import importlib.metadata
import json
import platform
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from deployment.mambo_deploy import Predictor
from deployment.mambo_deploy.bundle import Bundle
from deployment.mambo_deploy.download import default_bundle
from deployment.mambo_deploy.preprocessing import preprocess
from deployment.mambo_deploy.results import Prediction, hierarchy
from dev.benchmarks.inference.onnx_calibration import calibrate, load_batch
from dev.benchmarks.inference.onnx_inference import file_hash, model_files
from dev.benchmarks.inference.quality_compare import COLUMNS, compare
from dev.releases.mambo_v3.evaluation_data import canonical_rows, write_json


def select(rows, classes, calibration_count, evaluation_count, seed):
    """Bounded deterministic sampling of original train/test rows; never resplit."""
    queues = {"train": [], "test": []}
    limits = {"train": calibration_count, "test": evaluation_count}
    retained = {split: set() for split in queues}
    leaves = set(classes["labels"][0])
    for row in rows:
        split = "test" if int(row["set"]) == 0 else "train" if int(row["set"]) > 1 else "val"
        labels = [str(int(row[key])) for key in ("speciesKey", "genusKey", "familyKey")]
        if split not in queues or labels[0] not in leaves:
            continue
        filename = row["filename"]
        if Path(filename).name != filename or filename in ("", ".", ".."):
            raise ValueError(f"Expected a basename in filename: {filename!r}")
        path = f"images/{labels[0]}/{filename}"
        if path in retained[split]:
            continue
        priority = int(hashlib.sha256(f"{seed}:{split}:{path}".encode()).hexdigest(), 16)
        record = {"path": path, "labels": labels, "split": split}
        queue = queues[split]
        if len(queue) < limits[split]:
            heapq.heappush(queue, (-priority, path, record))
            retained[split].add(path)
        elif priority < -queue[0][0]:
            _, removed, _ = heapq.heapreplace(queue, (-priority, path, record))
            retained[split].remove(removed)
            retained[split].add(path)
    if any(len(queues[s]) != limits[s] for s in queues):
        raise ValueError("Not enough distinct known-species images in the original train/test splits")
    return {split: [item[2] for item in sorted(queue, reverse=True)] for split, queue in queues.items()}


def prepare(args):
    from mini_trainer.integrations.parquet import iter_parquet

    if min(args.calibration_count, args.evaluation_count, args.threads) < 1:
        raise ValueError("Counts and threads must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    bundle = Bundle(args.bundle or default_bundle(), download=not args.bundle)
    samples = select(iter_parquet(args.metadata), bundle.classes, args.calibration_count, args.evaluation_count, args.seed)
    seen = {}
    for split, records in samples.items():
        for record in records:
            image = (args.root / record["path"]).resolve()
            if not image.is_relative_to(args.root.resolve()):
                raise ValueError("Image escapes data root")
            record["sha256"] = file_hash(image)
            previous = seen.setdefault(record["sha256"], split)
            if previous != split:
                raise ValueError("Byte-identical images occur in calibration and evaluation selections")
    provenance = {
        "metadata_sha256": file_hash(args.metadata),
        "bundle_sha256": file_hash(bundle.root / "release.json"),
        "preprocessing": bundle.preprocessing,
        "seed": args.seed,
        "split_policy": "original set=0 test, set=1 unused validation, set>1 training; known species only",
    }
    inputs = args.output / "inputs"
    inputs.mkdir()
    batches = []
    for i, record in enumerate(samples["train"]):
        destination = inputs / f"{i:05d}.npz"
        np.savez(destination, images=preprocess(args.root / record["path"])[None])
        batches.append(
            {"path": destination.relative_to(args.output).as_posix(), "sample_ids": [record["path"]], "sha256": file_hash(destination)}
        )
    write_json(
        args.output / "calibration.json",
        {"schema_version": 1, "split": "train", "batch_input": "images", "provenance": provenance, "batches": batches},
    )
    write_json(args.output / "samples.json", samples)
    write_json(
        args.output / "prepared.json",
        {
            "bundle": str(bundle.root),
            "automatic_bundle": not args.bundle,
            "root": str(args.root.resolve()),
            "provenance": provenance,
            "samples_sha256": file_hash(args.output / "samples.json"),
            "calibration_sha256": file_hash(args.output / "calibration.json"),
            "source_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[3], text=True
            ).strip(),
            "runner_sha256": file_hash(__file__),
            "versions": {name: importlib.metadata.version(name) for name in ("torch", "torchao", "onnx", "onnxruntime", "numpy")},
        },
    )


def batches(root):
    manifest = json.loads((root / "calibration.json").read_text())
    for batch in manifest["batches"]:
        yield load_batch(root / "calibration.json", manifest, batch)[0]["images"]


def native_model(bundle):
    import torch

    from mini_trainer.deploy import Predictor as NativePredictor
    from mini_trainer.modeling import classification_module

    class Outputs(torch.nn.Module):
        def __init__(self, head):
            super().__init__()
            self.head = head

        def forward(self, features):
            return self.head(features)[0], self.head.preclassification(features)

    native = NativePredictor(device="cpu", weights=bundle.profile("torch"))
    setattr(native.model, native.model._backbone_output_name, Outputs(classification_module(native.model)))
    return native.model.eval()


def onnx_session(path, threads, *, profile=False, optimize=True, profile_prefix=None):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL if optimize else ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.enable_profiling = profile
    options.profile_file_prefix = str(profile_prefix or Path(path).parent / "execution")
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    session.disable_fallback()
    return session


def evaluate(args, bundle, samples, baseline, candidate, destination, provenance):
    selectors = {name: Predictor(bundle, model=name).selected for name in ("full", *bundle.regions)}
    rows = {name: {mode: [] for mode in ("baseline", "candidate")} for name in selectors}
    times = {mode: [] for mode in ("baseline", "candidate")}
    cosines = []
    first = preprocess(args.root / samples[0]["path"])[None]
    for _ in range(3):
        baseline(first)
        candidate(first)
    for i, record in enumerate(samples):
        image = args.root / record["path"]
        if file_hash(image) != record["sha256"]:
            raise ValueError("Evaluation image changed after preparation")
        image = preprocess(image)[None]
        outputs = {}
        order = (("baseline", baseline), ("candidate", candidate))
        for mode, run in order if i % 2 else reversed(order):
            started = time.perf_counter()
            leaf, embedding = run(image)
            times[mode].append(time.perf_counter() - started)
            if leaf.shape != (1, len(bundle.classes["labels"][0])) or embedding.shape != (1, bundle.manifest["embedding"]["dimension"]):
                raise ValueError("Unexpected score/embedding shape")
            if not np.isfinite(leaf).all() or not np.isfinite(embedding).all():
                raise ValueError("Nonfinite score/embedding output")
            outputs[mode] = embedding
            for name, selected in selectors.items():
                prediction = Prediction(*hierarchy(leaf, selected, bundle.classes))
                rows[name][mode].extend(row[: len(COLUMNS)] for row in canonical_rows([record], prediction, i))
        a, b = outputs.values()
        cosines.append(float(np.sum(a * b) / max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-12)))
    reports = {}
    for name, tables in rows.items():
        folder = destination / name
        folder.mkdir()
        manifest = {
            "schema_version": 1,
            "split": "test",
            "provenance": provenance,
            "levels": [
                {"name": rank, "classes": labels}
                for rank, labels in zip(("species", "genus", "family"), bundle.classes["labels"], strict=True)
            ],
            "samples": [{"instance_id": i, "filename": r["path"], "labels": r["labels"]} for i, r in enumerate(samples)],
        }
        for mode, table in tables.items():
            path = folder / f"{mode}.csv"
            with path.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(COLUMNS)
                writer.writerows(table)
            manifest[mode] = {
                "path": path.name,
                "sha256": file_hash(path),
                "classes": bundle.classes["labels"],
                "provenance": {**provenance, "backend": args.stage, "variant": mode},
            }
        write_json(folder / "comparison.json", manifest)
        reports[name] = compare(folder / "comparison.json", folder / "metrics")["levels"]
    return {
        "quality": reports,
        "embedding_cosine": {"minimum": min(cosines), "mean": statistics.mean(cosines)},
        "median_seconds": {mode: statistics.median(values) for mode, values in times.items()},
        "timing_scope": (
            "CPU batch-one forward, after three warmups, alternating order; excludes decode/preprocessing; not RSS or Space performance"
        ),
    }


def onnx_directory(args):
    return args.output / ("onnx" if args.calibration_method == "percentile" else "onnx-minmax")


def backend(args):
    import torch

    destination = onnx_directory(args) if args.stage == "onnx" else args.output / args.stage
    destination.mkdir(exist_ok=False)
    report = {
        "status": "running",
        "backend": args.stage,
        "release_ready": False,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "threads": args.threads,
        "coverage_target": "All Conv/Linear operations use integer kernels; other operations may remain floating point",
    }
    try:
        torch.set_num_threads(args.threads)
        torch.manual_seed(args.seed)
        prepared = json.loads((args.output / "prepared.json").read_text())
        for filename in ("samples", "calibration"):
            if file_hash(args.output / f"{filename}.json") != prepared[f"{filename}_sha256"]:
                raise ValueError("Prepared input manifest changed")
        args.root = Path(prepared["root"])
        bundle = Bundle(prepared["bundle"], download=prepared["automatic_bundle"])
        if file_hash(bundle.root / "release.json") != prepared["provenance"]["bundle_sha256"]:
            raise ValueError("Source bundle changed")
        report["provenance"] = prepared
        samples = json.loads((args.output / "samples.json").read_text())["test"]
        x = next(batches(args.output))
        if args.stage == "torch":
            from mini_trainer.modeling.quantization import load_int8, prepare_int8

            model = native_model(bundle)
            quantized = prepare_int8(model, torch.from_numpy(x))
            with torch.inference_mode():
                for batch in batches(args.output):
                    quantized(torch.from_numpy(batch))
            artifact = destination / "artifact"
            quantized.convert().save(artifact, torch.from_numpy(x), preprocessing=bundle.preprocessing, calibration=prepared)
            runtime, report["integer_execution"] = load_int8(artifact).lower(torch.from_numpy(x))

            def run(model, images):
                with torch.inference_mode():
                    return tuple(value.numpy() for value in model(torch.from_numpy(images)))

            def baseline(images):
                return run(model, images)

            def candidate(images):
                return run(runtime, images)

            report["artifact_files"] = [
                {"path": str(p), "sha256": file_hash(p), "bytes": p.stat().st_size} for p in artifact.iterdir() if p.is_file()
            ]
        else:
            source = bundle.profile("onnx-embedding")
            calibrate(
                source, args.output / "calibration.json", destination / "calibration", method=args.calibration_method, threads=args.threads
            )
            report["calibration_method"] = args.calibration_method
            path = destination / "calibration/model.onnx"
            session = onnx_session(path, args.threads, profile=True)
            session.run(None, {"images": x})
            profile = json.loads(Path(session.end_profiling()).read_text())
            ops = [event.get("args", {}) for event in profile if event.get("cat") == "Node"]
            integer = [op for op in ops if op.get("op_name", "").startswith(("QLinear", "MatMulInteger", "ConvInteger", "QGemm"))]
            if not integer or any(op.get("provider") != "CPUExecutionProvider" for op in integer):
                raise ValueError("No verified CPU integer execution in ONNX profile")
            floating = sorted(
                {op.get("op_name") for op in ops if op.get("op_name") in {"Conv", "FusedConv", "Gemm", "MatMul", "FusedMatMul"}}
            )
            if floating:
                raise ValueError(f"Incomplete INT8 coverage; floating weighted operations remain: {floating}")
            report["integer_execution"] = sorted({op["op_name"] for op in integer})
            report["floating_weighted_operations"] = floating
            reference = onnx_session(source, args.threads)

            def baseline(images):
                return reference.run(["output_0", "embedding"], {"images": images})

            def candidate(images):
                return session.run(["output_0", "embedding"], {"images": images})

            report["artifact_files"] = model_files(path, __import__("onnx"))
        report.update(evaluate(args, bundle, samples, baseline, candidate, destination, prepared["provenance"]))
        report.update(
            status="evaluated",
            next_step="Review quality/resource trade-offs; qualify batching, TTA and installed adapters before release/Space integration",
        )
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        raise
    finally:
        write_json(destination / "report.json", report)


def split_models(directory):
    """Cut existing graphs before the trained hidden layer, retaining QDQ scales."""
    import onnx

    calibration = directory / "calibration"
    evidence = json.loads((calibration / "report.json").read_text())
    for item in evidence["preprocessed_files"]:
        if file_hash(item["path"]) != item["sha256"]:
            raise ValueError("Preprocessed FP32 graph changed")
    destination = directory / "split"
    destination.mkdir(exist_ok=False)
    paths = {}
    boundary = None
    for mode, source in (("fp32", "preprocessed.onnx"), ("qdq", "model.onnx")):
        graph = onnx.load(calibration / source)
        if mode == "fp32":
            heads = [n for n in graph.graph.node if n.op_type == "Gemm" and n.input[1].endswith(".classifier.hidden.weight")]
            if len(heads) != 1:
                raise ValueError("Expected one Nemo hidden-layer Gemm to identify the backbone/head boundary")
            boundary = heads[0].input[0]
        extractor = onnx.utils.Extractor(graph)
        for part, inputs, outputs in (("backbone", ["images"], [boundary]), ("head", [boundary, "images"], ["output_0", "embedding"])):
            # Nemo also reads the original image shape when reshaping head outputs.
            model = extractor.extract_model(inputs, outputs)
            if part == "head" and any(node.op_type == "Conv" for node in model.graph.node):
                raise ValueError("Head extraction unexpectedly retained backbone convolutions")
            path = destination / f"{mode}-{part}.onnx"
            onnx.save_model(model, path, save_as_external_data=True, all_tensors_to_one_file=True, location=f"{mode}-{part}.data")
            paths[mode, part] = path
    return paths, boundary


def split_evaluate(args, directory, bundle, samples, images):
    paths, boundary = split_models(directory)
    sessions = {key: onnx_session(path, args.threads, optimize=False) for key, path in paths.items()}
    values = {}
    for backbone in ("fp32", "qdq"):
        features = [sessions[backbone, "backbone"].run([boundary], {"images": image})[0] for image in images]
        for head in ("fp32", "qdq"):
            rows = [
                sessions[head, "head"].run(["output_0", "embedding"], {boundary: feature, "images": image})
                for feature, image in zip(features, images, strict=True)
            ]
            values[f"{backbone}_backbone_{head}_head"] = [np.concatenate([row[i] for row in rows]) for i in (0, 1)]
    # Check each split control against its intact graph on one identical input.
    for mode, source in (("fp32", "preprocessed.onnx"), ("qdq", "model.onnx")):
        intact = onnx_session(directory / "calibration" / source, args.threads, optimize=False)
        expected = intact.run(["output_0", "embedding"], {"images": images[0]})
        for actual, reference in zip(values[f"{mode}_backbone_{mode}_head"], expected, strict=True):
            np.testing.assert_allclose(actual[:1], reference, rtol=1e-4, atol=1e-4, err_msg="Graph split changed its control")
        del intact
    reference_scores, reference_embeddings = values["fp32_backbone_fp32_head"]
    reference_predictions = reference_scores.argmax(axis=1)
    results = {}
    for name, (scores, embeddings) in values.items():
        if (
            scores.shape != reference_scores.shape
            or embeddings.shape != reference_embeddings.shape
            or not np.isfinite(scores).all()
            or not np.isfinite(embeddings).all()
        ):
            raise ValueError(f"Invalid split outputs: {name}")
        predictions = scores.argmax(axis=1)
        cosine = np.sum(embeddings * reference_embeddings, axis=1) / np.maximum(
            np.linalg.norm(embeddings, axis=1) * np.linalg.norm(reference_embeddings, axis=1), 1e-12
        )
        results[name] = {
            "correct_species": sum(bundle.classes["labels"][0][i] == r["labels"][0] for i, r in zip(predictions, samples, strict=True)),
            "predictions_agree_with_fp32": int(np.sum(predictions == reference_predictions)),
            "embedding_cosine_mean": float(cosine.mean()),
            "embedding_cosine_min": float(cosine.min()),
        }
    report = {
        "samples": len(samples),
        "sample_paths": [r["path"] for r in samples],
        "calibration_method": args.calibration_method,
        "boundary": boundary,
        "results": results,
        "scope": (
            "Existing QDQ weights/scales, optimizations disabled; head includes trained hidden layer, normalization and final classifier; "
            "first-image split controls verified; no recalibration"
        ),
    }
    write_json(directory / "split/report.json", report)
    print(json.dumps(report, indent=2))


def diagnose(args):
    """Compare existing FP32, unoptimized QDQ and optimized INT8 on held-out images."""
    prepared = json.loads((args.output / "prepared.json").read_text())
    if file_hash(args.output / "samples.json") != prepared["samples_sha256"]:
        raise ValueError("Prepared sample manifest changed")
    bundle = Bundle(prepared["bundle"], download=prepared["automatic_bundle"])
    if file_hash(bundle.root / "release.json") != prepared["provenance"]["bundle_sha256"]:
        raise ValueError("Source bundle changed")
    directory = onnx_directory(args)
    previous = json.loads((directory / "report.json").read_text())
    for item in previous["artifact_files"]:
        if file_hash(item["path"]) != item["sha256"]:
            raise ValueError("Quantized artifact changed")
    samples = json.loads((args.output / "samples.json").read_text())["test"][:32]
    images = []
    for record in samples:
        path = Path(prepared["root"]) / record["path"]
        if file_hash(path) != record["sha256"]:
            raise ValueError("Evaluation image changed after preparation")
        images.append(preprocess(path)[None])
    if args.stage == "split":
        split_evaluate(args, directory, bundle, samples, images)
        return
    candidate = directory / "calibration/model.onnx"
    results, outputs = {}, {}
    for name, path, optimize in (("fp32", bundle.profile("onnx-embedding"), True), ("qdq", candidate, False), ("int8", candidate, True)):
        session = onnx_session(
            path, args.threads, profile=True, optimize=optimize, profile_prefix=args.output / f"{directory.name}-diagnostic-{name}"
        )
        values = [session.run(["output_0", "embedding"], {"images": image}) for image in images]
        scores, embeddings = [np.concatenate([value[i] for value in values]) for i in (0, 1)]
        if (
            scores.shape != (len(samples), len(bundle.classes["labels"][0]))
            or embeddings.shape != (len(samples), bundle.manifest["embedding"]["dimension"])
            or not np.isfinite(scores).all()
            or not np.isfinite(embeddings).all()
        ):
            raise ValueError(f"Invalid {name} score/embedding outputs")
        predictions = scores.argmax(axis=1)
        profile = json.loads(Path(session.end_profiling()).read_text())
        operators = sorted({event.get("args", {}).get("op_name", "") for event in profile if event.get("cat") == "Node"})
        integer = [op for op in operators if op.startswith(("QLinear", "MatMulInteger", "ConvInteger", "QGemm"))]
        results[name] = {
            "correct_species": sum(
                bundle.classes["labels"][0][i] == record["labels"][0] for i, record in zip(predictions, samples, strict=True)
            ),
            "integer_operators": integer,
            "operators": operators,
        }
        outputs[name] = predictions, embeddings
        del session
    for name in ("qdq", "int8"):
        predictions, embeddings = outputs[name]
        reference, original = outputs["fp32"]
        cosine = np.sum(embeddings * original, axis=1) / np.maximum(
            np.linalg.norm(embeddings, axis=1) * np.linalg.norm(original, axis=1), 1e-12
        )
        results[name].update(
            predictions_agree_with_fp32=int(np.sum(predictions == reference)),
            embedding_cosine_mean=float(cosine.mean()),
            embedding_cosine_min=float(cosine.min()),
        )
    report = {
        "samples": len(samples),
        "sample_paths": [r["path"] for r in samples],
        "threads": args.threads,
        "results": results,
        "qdq_int8_prediction_agreement": int(np.sum(outputs["qdq"][0] == outputs["int8"][0])),
        "unoptimized_qdq_has_integer_kernels": bool(results["qdq"]["integer_operators"]),
        "artifact_files": previous["artifact_files"],
        "samples_sha256": prepared["samples_sha256"],
        "scope": "First 32 prepared held-out images; global species accuracy and embeddings; no recalibration or performance benchmark",
    }
    write_json(args.output / f"{directory.name}-diagnostic.json", report)
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--root", type=Path, help="Directory containing images/SPECIES/FILENAME")
    parser.add_argument("--output", type=Path, required=True, help="New persistent output directory")
    parser.add_argument("--bundle", type=Path, help="Verified FP32 bundle; omitted uses the pinned automatic cache")
    parser.add_argument("--calibration-count", type=int, default=128)
    parser.add_argument("--evaluation-count", type=int, default=256)
    parser.add_argument(
        "--calibration-method",
        choices=("percentile", "minmax"),
        default="percentile",
        help="ONNX calibration recipe; MinMax writes onnx-minmax/ beside the original onnx/ result",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--stage", choices=("all", "prepare", "torch", "onnx", "diagnose", "split"), default="all")
    args = parser.parse_args()
    if args.stage in ("diagnose", "split"):
        diagnose(args)
        return
    if args.stage in ("all", "prepare") and (args.metadata is None or args.root is None):
        parser.error("--metadata and --root are required for input preparation")
    if args.stage in ("all", "prepare"):
        prepare(args)
    if args.stage in ("torch", "onnx"):
        backend(args)
    elif args.stage == "all":
        summary = {"release_ready": False, "backends": {}}
        for name in ("torch", "onnx"):
            command = [sys.executable, "-m", "dev.releases.mambo_v3.quantize", *sys.argv[1:], "--stage", name]
            with (args.output / f"{name}.log").open("w") as stream:
                code = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT).returncode
            folder = onnx_directory(args).name if name == "onnx" else name
            summary["backends"][name] = {"exit_code": code, "report": f"{folder}/report.json"}
            write_json(args.output / "summary.json", summary)
        if any(item["exit_code"] for item in summary["backends"].values()):
            raise SystemExit("At least one backend failed; inspect summary.json and backend logs. Nothing was published.")


if __name__ == "__main__":
    main()
