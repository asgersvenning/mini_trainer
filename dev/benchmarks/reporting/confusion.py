"""Fresh-process dense soft-confusion rendering benchmark, including artifact I/O."""

import argparse
import importlib.util
import json
import resource
import time
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from mini_trainer.logging.confusion import confusion_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classes", type=int, default=5000)
    parser.add_argument("--baseline", type=Path, help="Previous visualization/plot.py saved from Git")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    n = args.classes
    rng = np.random.default_rng(42)
    matrix = np.empty((n, n), dtype=np.float32)
    columns = np.arange(n)
    for start in range(0, n, 64):
        rows = np.arange(start, min(n, start + 64))
        values = rng.random((len(rows), n), dtype=np.float32) * 0.01
        values += (rows[:, None] // max(1, n // 20) == columns[None, :] // max(1, n // 20)) * 0.2
        values[np.arange(len(rows)), rows] += 5
        matrix[rows] = values / values.sum(axis=1, keepdims=True)
    if args.baseline:
        spec = importlib.util.spec_from_file_location("heatmap_baseline", args.baseline)
        baseline = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(baseline)
    started = time.perf_counter()
    if args.baseline:
        preview = baseline.plot_heatmap(matrix)
    else:
        preview = confusion_report(matrix, soft=True, directory=args.output)
    plt.imsave(args.output / "dashboard.png", preview)
    result = {
        "classes": n,
        "baseline": str(args.baseline) if args.baseline else None,
        "seconds": time.perf_counter() - started,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "dashboard_shape": list(preview.shape),
        "artifact_bytes": {p.name: p.stat().st_size for p in args.output.iterdir()},
        "scope": "CPU rendering and PNG/data export; RSS includes imports/input; excludes DDP and remote upload",
    }
    (args.output / "measurement.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
