"""Stream three composed TTA policies from seven shared views per image."""

import argparse
import csv
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from functools import partial
from pathlib import Path

import numpy as np

from deployment.mambo_deploy import TTA, Predictor
from deployment.mambo_deploy.preprocessing import _rgb, preprocess
from deployment.mambo_deploy.results import Prediction, hierarchy
from dev.benchmarks.inference.onnx_inference import file_hash
from dev.releases.mambo_v3.compact_tta import policies
from dev.releases.mambo_v3.evaluate import runtime_settings
from dev.releases.mambo_v3.evaluation_data import CSV_COLUMNS, canonical_rows, load_records, write_json

SHORTLIST = ("rotation30_pad15_3", "rotation30_pad25_3", "wide_rotation_mixed_padding_5")


def prepare(image, transform):
    return preprocess(transform(image.copy()))


def collect(args):
    args.output.mkdir(parents=True, exist_ok=False)
    _, records = load_records(args.manifest, args.root, args.count, 20260923)
    views, parameters, all_recipes = policies()
    recipes = {name: all_recipes[name] for name in SHORTLIST}
    keys = list(dict.fromkeys(key for names in recipes.values() for key in names))
    report = {
        "status": "running",
        "backend": args.backend,
        "samples": len(records),
        "processed": 0,
        "recipes": recipes,
        "parameters": {key: parameters[key] for key in keys},
        "manifest_sha256": file_hash(args.manifest),
        "bundle_sha256": file_hash(args.bundle / "release.json"),
        "runner_sha256": file_hash(__file__),
        "transforms_sha256": file_hash(Path(__file__).with_name("compact_tta.py")),
        "runtime": runtime_settings(4, args.backend),
        "batch_size": args.batch_size,
    }
    predictor = Predictor(args.bundle, backend=args.backend, device="cuda:0", model="north_europe", threads=4, batch_size=args.batch_size)
    report["effective_precision"] = predictor.effective_precision
    start = time.perf_counter()
    write_json(args.output / "report.json", report)
    try:
        with ExitStack() as stack:
            pool = stack.enter_context(ThreadPoolExecutor(max_workers=4))
            writers = {}
            for name in recipes:
                folder = args.output / name
                folder.mkdir()
                stream = stack.enter_context((folder / "mini_metric.csv").open("w", newline=""))
                writers[name] = csv.writer(stream)
                writers[name].writerow(CSV_COLUMNS)
            for offset in range(0, len(records), args.batch_size):
                batch = records[offset : offset + args.batch_size]
                paths = [args.root / r["path"] for r in batch]
                for path, record in zip(paths, batch, strict=True):
                    if file_hash(path) != record["sha256"]:
                        raise ValueError(f"Changed image: {path}")
                decoded = list(pool.map(_rgb, paths))
                raw = {}
                for key in keys:
                    prepared = np.stack(list(pool.map(partial(prepare, transform=views[key]), decoded)))
                    raw[key] = predictor._infer(prepared, False)[0].astype(np.float32)
                    if not np.isfinite(raw[key]).all():
                        raise ValueError("Nonfinite logits")
                for name, names in recipes.items():
                    averaged = sum(raw[key] / np.float32(len(names)) for key in names)
                    prediction = Prediction(*hierarchy(averaged, predictor.selected, predictor.bundle.classes))
                    writers[name].writerows(canonical_rows(batch, prediction, offset))
                    if offset == 0:
                        predictor.tta = TTA(tuple(views[key] for key in names), name)
                        check = predictor._infer_batch(paths, pool=pool)[0]
                        np.testing.assert_allclose(check, averaged, rtol=0, atol=1e-6)
                report["processed"] = offset + len(batch)
                if offset % (args.batch_size * 25) == 0:
                    report["elapsed_seconds"] = time.perf_counter() - start
                    write_json(args.output / "report.json", report)
                    print(args.backend, report["processed"], len(records), round(report["elapsed_seconds"], 1), flush=True)
        report.update(status="complete", csv_sha256={n: file_hash(args.output / n / "mini_metric.csv") for n in recipes})
        if args.backend == "onnx":
            report["providers"] = {k: s.get_providers() for k, s in predictor._sessions.items()}
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - start
        write_json(args.output / "report.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "manifest", "root", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--backend", choices=("torch", "onnx"), required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--count", type=int)
    collect(parser.parse_args())
