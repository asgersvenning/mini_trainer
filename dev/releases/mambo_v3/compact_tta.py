"""Bounded, compositional padding/rotation TTA qualification; no default changes."""

import argparse
import csv
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
from PIL import Image

from deployment.mambo_deploy import TTA, EdgePad, Predictor, View
from deployment.mambo_deploy.results import Prediction, hierarchy
from dev.releases.mambo_v3.evaluate import runtime_settings
from dev.releases.mambo_v3.evaluation_data import CSV_COLUMNS, canonical_rows, write_json


def rotate_pad(image, degrees, padding):
    """Rotate on an expanded canvas once, then edge-pad before ordinary preprocessing."""
    rotated = Image.fromarray(image.transpose(1, 2, 0)).rotate(
        degrees, resample=Image.Resampling.BILINEAR, expand=True, fillcolor=(124, 116, 104)
    )
    return EdgePad(padding)(np.asarray(rotated).transpose(2, 0, 1))


def policies():
    views = {"original": View(), "pad08": EdgePad(0.08), "pad15": EdgePad(0.15)}
    parameters = {"original": {}, "pad08": {"padding": 0.08}, "pad15": {"padding": 0.15}}
    for degrees, padding in [(d, p) for d in (-10, 10, -30, 30) for p in (0.08, 0.15)] + [(-30, 0.25), (30, 0.25)]:
        name = f"rot{degrees}_pad{padding}"
        parameters[name] = {"degrees": degrees, "padding": padding}
        views[name] = partial(rotate_pad, **parameters[name])
    recipes = {
        "none": ["original"],
        "padded_scale": ["original", "pad08", "pad15"],
        "wide_rotation_5": ["original", *[f"rot{d}_pad0.08" for d in (-10, 10, -30, 30)]],
        "wide_rotation_pad15_5": ["original", *[f"rot{d}_pad0.15" for d in (-10, 10, -30, 30)]],
        "wide_rotation_mixed_padding_5": ["original", "rot-10_pad0.15", "rot10_pad0.15", "rot-30_pad0.25", "rot30_pad0.25"],
        "rotation10_3": ["original", "rot-10_pad0.08", "rot10_pad0.08"],
        "rotation30_3": ["original", "rot-30_pad0.08", "rot30_pad0.08"],
        "rotation10_pad15_3": ["original", "rot-10_pad0.15", "rot10_pad0.15"],
        "rotation30_pad15_3": ["original", "rot-30_pad0.15", "rot30_pad0.15"],
        "rotation30_pad25_3": ["original", "rot-30_pad0.25", "rot30_pad0.25"],
        "mixed_3": ["original", "rot-10_pad0.08", "rot30_pad0.15"],
        "mixed_mirrored_3": ["original", "rot10_pad0.08", "rot-30_pad0.15"],
    }
    return views, parameters, recipes


def run(args):
    if args.output.exists():
        raise ValueError("Use a fresh output directory")
    args.output.mkdir(parents=True)
    samples = json.loads(args.samples.read_text())
    reporting = json.loads(args.reporting_ids.read_text())
    candidates = sorted(set(reporting) - set(samples["ids"]))
    random_ids = sorted(map(int, np.random.default_rng(20260925).choice(candidates, args.count, replace=False)))
    ids = samples["ids"] + random_ids
    manifest = json.loads(args.manifest.read_text())
    records = [manifest["records"][i] for i in ids]
    paths = [args.root / r["path"] for r in records]
    for path, record in zip(paths, records, strict=True):
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"Changed image: {path}")
    views, parameters, recipes = policies()
    settings = runtime_settings(4, "torch")
    predictor = Predictor(args.bundle, backend="torch", device="cuda:0", model="north_europe", batch_size=32, threads=4)
    splits = {
        "focus": [0, len(samples["focus"])],
        "controls": [len(samples["focus"]), len(samples["ids"])],
        "random": [len(samples["ids"]), len(ids)],
    }
    report = {
        "inputs_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (args.samples, args.reporting_ids, args.manifest)
        },
        "status": "running",
        "ids": ids,
        "records": records,
        "splits": splits,
        "seed": 20260925,
        "runtime": settings,
        "parameters": parameters,
        "recipes": recipes,
        "seconds_per_view": {},
        "selection": "Existing flagged cases and controls plus a disjoint uniform random reporting sample; exploratory only",
    }
    write_json(args.output / "report.json", report)
    logits = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for name, transform in views.items():
            predictor.tta = TTA((transform,), name)
            start = time.monotonic()
            logits[name] = np.concatenate([predictor._infer_batch(paths[i : i + 32], pool=pool)[0] for i in range(0, len(paths), 32)])
            report["seconds_per_view"][name] = time.monotonic() - start
            print(name, round(report["seconds_per_view"][name], 2), flush=True)
        report["effective_precision"] = predictor.effective_precision
        for name, keys in recipes.items():
            raw = sum(logits[key] / np.float32(len(keys)) for key in keys)
            folder = args.output / name
            folder.mkdir()
            for subset, (start, end) in splits.items():
                pred = Prediction(*hierarchy(raw[start:end], predictor.selected, predictor.bundle.classes))
                with (folder / f"{subset}.csv").open("w", newline="") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(CSV_COLUMNS)
                    writer.writerows(canonical_rows(records[start:end], pred))
            if name == "rotation30_pad15_3":
                predictor.tta = TTA(tuple(views[k] for k in keys), name)
                actual = predictor._infer_batch(paths[:32], pool=pool)[0]
                np.testing.assert_allclose(actual, raw[:32], rtol=0, atol=1e-6)
    np.savez_compressed(args.output / "view_logits.npz", **logits)
    report["aggregation_check"] = "Normal TTA API agrees at atol=1e-6 with identical batch shape"
    report["status"] = "complete"
    write_json(args.output / "report.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--reporting-ids", type=Path, required=True)
    parser.add_argument("--count", type=int, default=1024)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
