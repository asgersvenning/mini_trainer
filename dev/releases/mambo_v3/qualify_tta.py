"""Bounded real-image qualification of outer TTA; write canonical mini_metrics inputs."""

import argparse
import csv
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from deployment.mambo_deploy import Predictor
from deployment.mambo_deploy.augmentation import PROFILES, resolve_tta
from deployment.mambo_deploy.results import Prediction, hierarchy
from dev.benchmarks.inference.onnx_inference import file_hash
from dev.releases.mambo_v3.benchmark import timing
from dev.releases.mambo_v3.evaluate import runtime_settings
from dev.releases.mambo_v3.evaluation_data import CSV_COLUMNS, PRESETS, canonical_rows, load_records, write_json
from dev.releases.mambo_v3.tta_candidates import CANDIDATES, candidate_policy


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    _, records = load_records(args.manifest, args.root, args.count, 20260923)
    paths = [args.root / r["path"] for r in records]
    write_json(args.output / "samples.json", records)
    report = {
        "status": "running",
        "backend": args.backend,
        "samples": len(records),
        "sample_ids_sha256": file_hash(args.output / "samples.json"),
        "runtime": runtime_settings(4, args.backend),
        "profiles": {},
        "bundle_sha256": file_hash(args.bundle / "release.json"),
        "runner_sha256": file_hash(__file__),
    }
    try:
        if any(file_hash(path) != r["sha256"] for path, r in zip(paths, records, strict=True)):
            raise ValueError("Image bytes changed")
        p = Predictor(args.bundle, backend=args.backend, device="cuda:0", model="full", threads=4, batch_size=32)
        selectors = {name: Predictor(args.bundle, model=name).selected for name in PRESETS}
        for profile in args.profiles:
            p.tta = candidate_policy(profile) if profile in CANDIDATES else resolve_tta(profile)
            start = time.perf_counter()
            outputs = []
            with ThreadPoolExecutor(max_workers=4) as pool:
                for offset in range(0, len(paths), 32):
                    scores, _ = p._infer_batch(paths[offset : offset + 32], pool=pool)
                    if scores.dtype != np.float32 or not np.isfinite(scores).all():
                        raise AssertionError("Invalid leaf scores")
                    outputs.append(scores)
            raw = np.concatenate(outputs)
            folder = args.output / profile
            folder.mkdir()
            hashes = {}
            for preset, indices in selectors.items():
                result = Prediction(*hierarchy(raw, indices, p.bundle.classes))
                destination = folder / preset
                destination.mkdir()
                csv_path = destination / "mini_metric.csv"
                with csv_path.open("w", newline="") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(CSV_COLUMNS)
                    writer.writerows(canonical_rows(records, result))
                hashes[preset] = file_hash(csv_path)
            # Public API: bounded batches, custom mask, same mean embedding independent of mask.
            p._apply_class_mask(-1)
            plain = p.predict(paths[:8])
            embedded, vectors = p.predict_with_embeddings(paths[:8])
            assert plain.labels == embedded.labels
            assert vectors.shape == (8, 1280) and vectors.dtype == np.float32 and np.isfinite(vectors).all()
            np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4)
            custom = [p.bundle.classes["labels"][0][i] for i in selectors["north_europe"][:3]]
            p._apply_class_mask(custom)
            limited, masked_vectors = p.predict_with_embeddings(paths[:8])
            assert all(item.label[0] in custom for item in limited)
            np.testing.assert_array_equal(vectors, masked_vectors)
            p._apply_class_mask(-1)
            report["profiles"][profile] = {
                "csv_sha256": hashes,
                "elapsed_seconds": time.perf_counter() - start,
                "prediction_embedding_agreement_first8": True,
                "custom_list_first8": True,
                "finite_leaf_scores": True,
                "normalized_embeddings_first8": True,
            }
            write_json(args.output / "report.json", report)
            print(profile, report["profiles"][profile]["elapsed_seconds"], flush=True)
        report["status"] = "complete"
    except Exception as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        write_json(args.output / "report.json", report)


def benchmark(args):
    args.output.mkdir(parents=True, exist_ok=False)
    runtime = runtime_settings(4, args.backend)
    _, records = load_records(args.manifest, args.root, 32, 20260923)
    paths = [args.root / r["path"] for r in records]
    if any(file_hash(path) != r["sha256"] for path, r in zip(paths, records, strict=True)):
        raise ValueError("Image bytes changed")
    p = Predictor(args.bundle, backend=args.backend, device="cuda:0", model="north_europe", threads=4, batch_size=32)
    report = {
        "status": "running",
        "backend": args.backend,
        "runtime": runtime,
        "samples": records,
        "cells": [],
        "purpose": "one process; one warmup and three observations per profile; complete CPU results",
        "bundle_sha256": file_hash(args.bundle / "release.json"),
        "runner_sha256": file_hash(__file__),
    }
    try:
        for profile in args.profiles:
            p.tta = candidate_policy(profile) if profile in CANDIDATES else resolve_tta(profile)
            p.predict(paths)
            measured = timing(lambda: p.predict(paths), 3)
            report["cells"].append(
                {"profile": profile, "batch": len(paths), "images_per_second": len(paths) / measured["median_seconds"], **measured}
            )
            print(profile, report["cells"][-1]["images_per_second"], flush=True)
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
    parser.add_argument("--backend", choices=["torch", "onnx"], required=True)
    parser.add_argument("--count", type=int, default=1024)
    parser.add_argument("--profiles", nargs="+", choices=(*PROFILES, *CANDIDATES), default=list(PROFILES))
    parser.add_argument("--timing-only", action="store_true")
    args = parser.parse_args()
    (benchmark if args.timing_only else run)(args)
