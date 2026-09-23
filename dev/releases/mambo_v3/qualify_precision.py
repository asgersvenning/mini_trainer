"""Bounded precision qualification: existing weights, all presets, FP32 output contracts."""

import argparse
import csv
import importlib.util
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from deployment.mambo_deploy import Predictor
from deployment.mambo_deploy.preprocessing import preprocess
from deployment.mambo_deploy.results import Prediction, hierarchy
from dev.benchmarks.inference.onnx_inference import file_hash
from dev.releases.mambo_v3.evaluate import prepare_batch, runtime_settings
from dev.releases.mambo_v3.evaluation_data import CSV_COLUMNS, PRESETS, canonical_rows, load_records, write_json


def qualify(args):
    args.output.mkdir(parents=True, exist_ok=False)
    runtime_settings(4, args.backend)
    _, records = load_records(args.manifest, args.root, args.count, 20260923)
    write_json(args.output / "samples.json", records)
    report = {"status": "running", "variants": {}, "samples_sha256": file_hash(args.output / "samples.json")}
    shared = None
    try:
        paths = [args.root / r["path"] for r in records]
        if any(file_hash(p) != r["sha256"] for p, r in zip(paths, records, strict=True)):
            raise ValueError("Image bytes changed")
        if args.old_preprocessing:
            spec = importlib.util.spec_from_file_location("old_recipe", args.old_preprocessing)
            old = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(old)
            for path in paths[:256]:
                np.testing.assert_array_equal(preprocess(path), old.preprocess(path))
            report["preprocessing_exact_images"] = min(256, len(paths))
        for precision in args.precisions:
            start = time.perf_counter()
            predictor = Predictor(
                args.bundle, backend=args.backend, device="cuda:0", model="full", batch_size=32, threads=4, precision=precision
            )
            if shared is not None:
                predictor._torch_model = shared
            runtime = predictor._torch if args.backend == "torch" else predictor._onnx
            with ThreadPoolExecutor(max_workers=4) as pool:
                outputs, vectors = [], []
                for offset in range(0, len(paths), 32):
                    x = prepare_batch(paths[offset : offset + 32], pool)
                    leaves, embedding = runtime(x, True)
                    assert leaves.dtype == embedding.dtype == np.float32
                    assert embedding.shape == (len(x), 1280)
                    assert np.isfinite(leaves).all() and np.isfinite(embedding).all()
                    np.testing.assert_allclose(np.linalg.norm(embedding, axis=1), 1, atol=1e-4)
                    if offset < 256:
                        plain, _ = runtime(x, False)
                        np.testing.assert_array_equal(np.argmax(plain, axis=1), np.argmax(leaves, axis=1))
                    outputs.append(leaves)
                    vectors.append(embedding)
            raw = np.concatenate(outputs)
            destination = args.output / precision
            destination.mkdir()
            np.save(destination / "embeddings.npy", np.concatenate(vectors), allow_pickle=False)
            for preset in PRESETS:
                selector = Predictor(args.bundle, model=preset)
                result = Prediction(*hierarchy(raw, selector.selected, selector.bundle.classes))
                folder = destination / preset
                folder.mkdir()
                with (folder / "mini_metric.csv").open("w", newline="") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(CSV_COLUMNS)
                    writer.writerows(canonical_rows(records, result))
            # Public API, custom-list filtering, threaded preparation and embedding path.
            predictor._apply_class_mask(selector.class_list[:3])
            public, embedding = predictor.predict_with_embeddings(paths[:8])
            assert all(item.label[0] in selector.class_list[:3] for item in public)
            assert public.labels == predictor.predict(paths[:8]).labels
            _, direct_embedding = runtime(prepare_batch(paths[:8]), True)
            np.testing.assert_array_equal(embedding, direct_embedding)
            shared = predictor._torch_model
            report["variants"][precision] = {
                "seconds": time.perf_counter() - start,
                "finite": True,
                "embedding_dtype": "float32",
                "prediction_modes_agree_first256": True,
                "custom_list": True,
            }
            write_json(args.output / "report.json", report)
            print(precision, report["variants"][precision], flush=True)
        report["status"] = "complete"
    except Exception as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        write_json(args.output / "report.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "manifest", "root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--old-preprocessing", type=Path)
    parser.add_argument("--backend", choices=["torch", "onnx"], required=True)
    parser.add_argument("--precisions", nargs="+", required=True)
    parser.add_argument("--count", type=int, default=4096)
    qualify(parser.parse_args())
