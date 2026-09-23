"""Separate worker scaling, prepared inference and bounded one-batch lookahead."""

import argparse
import itertools
import statistics
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from deployment.mambo_deploy import Predictor
from deployment.mambo_deploy.results import Prediction, hierarchy
from dev.benchmarks.inference.onnx_inference import file_hash
from dev.releases.mambo_v3.benchmark import snapshot, timing
from dev.releases.mambo_v3.evaluate import runtime_settings
from dev.releases.mambo_v3.evaluation_data import load_records, write_json


def stream(predictor, paths, workers, batch, overlap):
    """Experimental pipeline only: at most current and next prepared batch."""
    chunks = [paths[i : i + batch] for i in range(0, len(paths), batch)]
    leaves = []
    with ThreadPoolExecutor(max_workers=workers) as pool, ThreadPoolExecutor(max_workers=1) as producer:
        pending = producer.submit(predictor._prepare, chunks[0], pool) if overlap else None
        for i, chunk in enumerate(chunks):
            images = pending.result() if pending is not None else predictor._prepare(chunk, pool)
            pending = producer.submit(predictor._prepare, chunks[i + 1], pool) if overlap and i + 1 < len(chunks) else None
            leaves.append(predictor._infer(images)[0])
    return Prediction(*hierarchy(np.concatenate(leaves), predictor.selected, predictor.bundle.classes))


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    _, records = load_records(args.manifest, args.root, 128, 20260923)
    paths = [args.root / r["path"] for r in records]
    if any(file_hash(p) != r["sha256"] for p, r in zip(paths, records, strict=True)):
        raise ValueError("Image bytes changed")
    report = {
        "status": "running",
        "backend": args.backend,
        "runtime": runtime_settings(4, args.backend),
        "before": snapshot(),
        "samples": records,
        "cells": [],
        "streams": [],
        "bundle_sha256": file_hash(args.bundle / "release.json"),
        "runner_sha256": file_hash(__file__),
    }
    try:
        p = Predictor(args.bundle, backend=args.backend, device="cuda:0", model="north_europe", threads=4, batch_size=64)
        p.predict(paths[:1])
        report["precision"] = p.effective_precision
        configs = list(itertools.product((8, 32, 64), (1, 2, 4, 8)))
        for trial in range(3):
            for batch, workers in configs if trial != 1 else reversed(configs):
                p.preprocess_workers = workers
                prepared = p._prepare(paths[:batch])
                # Warm every boundary; inference returns CPU scores, so timing synchronizes.
                p._infer(prepared)
                p.predict(paths[:batch])
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    preparation = timing(lambda: p._prepare(paths[:batch], pool), 3)
                cell = {
                    "trial": trial,
                    "batch": batch,
                    "workers": workers,
                    "preparation": preparation,
                    "prepared": timing(lambda: p._infer(prepared), 3),
                    "end_to_end": timing(lambda: p.predict(paths[:batch]), 3),
                }
                report["cells"].append(cell)
                write_json(args.output / "report.json", report)
                print(trial, batch, workers, round(batch / cell["end_to_end"]["median_seconds"], 1), flush=True)
        expected = None
        for workers in (1, 4, 8):
            for overlap in (False, True):
                result = stream(p, paths, workers, 32, overlap)
                if expected is None:
                    expected = result.labels
                if result.labels != expected:
                    raise AssertionError("Worker/prefetch scheduling changed predictions")
                times = timing(lambda: stream(p, paths, workers, 32, overlap), 5)
                report["streams"].append({"workers": workers, "overlap": overlap, "images": len(paths), **times})
                print("stream", workers, overlap, round(len(paths) / statistics.median(times["seconds"]), 1), flush=True)
        report.update(status="complete", after=snapshot())
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
    run(parser.parse_args())
