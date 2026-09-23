import argparse
import inspect
import json
import statistics
import time
from pathlib import Path

import numpy as np

from deployment.mambo_deploy.preprocessing import RECIPE, _rgb, preprocess

parser = argparse.ArgumentParser(description="Diagnostic-only layout/crop interventions with exact pixel checks")
parser.add_argument("--evidence", type=Path, required=True)
parser.add_argument("--root", type=Path, required=True)
args = parser.parse_args()
root = args.evidence
records = json.loads((root / "samples.json").read_text())
paths = [args.root / r["path"] for r in records]
source = inspect.getsource(preprocess)
variants = {"baseline": preprocess}
for contiguous, crop in ((True, False), (False, True), (True, True)):
    s = source
    if crop:
        s = s.replace(
            "lo = np.floor(coordinates).astype(int)",
            "coordinates = coordinates[(resized-size)//2:(resized-size)//2+size]\n    lo = np.floor(coordinates).astype(int)",
        ).replace("offset = (resized - size) // 2", "offset = 0")
    if contiguous:
        s = (
            s.replace(
                "image = image[:, yy][:, :, xx].astype(np.float32)",
                "image = np.ascontiguousarray(image[:, yy][:, :, xx], dtype=np.float32)",
            )
            .replace("pixels = rows[:, :, lo]", "rows = np.ascontiguousarray(rows)\n    pixels = rows[:, :, lo]")
            .replace("    return np.ascontiguousarray(", "    pixels = np.ascontiguousarray(pixels)\n    return np.ascontiguousarray(")
        )
    env = {"np": np, "_rgb": _rgb, "RECIPE": RECIPE}
    exec(s, env)
    variants[f"contiguous-{contiguous}-crop-{crop}"] = env["preprocess"]
references = [preprocess(p) for p in paths]
out = []
for name, fn in variants.items():
    for p, ref in zip(paths, references):
        np.testing.assert_array_equal(fn(p), ref)
for trial in range(3):
    for name, fn in list(variants.items()) if trial != 1 else reversed(list(variants.items())):
        vals = []
        for _ in range(7):
            t = time.perf_counter()
            np.stack([fn(p) for p in paths])
            vals.append((time.perf_counter() - t) * 1000)
        out.append({"trial": trial, "variant": name, "ms": vals, "median_ms": statistics.median(vals), "byte_identical_32": True})
        print(out[-1], flush=True)
(root / "preprocess-interventions.json").write_text(json.dumps(out, indent=2))
