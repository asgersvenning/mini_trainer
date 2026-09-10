"""Fresh-process CPU benchmark for complete dendrogram rendering and SVG export."""

import argparse
import importlib.util
import json
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rc_context
from scipy.spatial.distance import pdist, squareform

import mini_trainer.visualization.dendrogram as current


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classes", type=int, default=3422)
    parser.add_argument("--deep", action="store_true")
    parser.add_argument("--module-file", type=Path, help="Optional baseline dendrogram.py saved from Git")
    parser.add_argument("--output", type=Path, required=True, help="New directory for SVG and measurement JSON")
    args = parser.parse_args()
    if args.classes < 2:
        parser.error("--classes must be at least 2")
    module = current
    if args.module_file:
        spec = importlib.util.spec_from_file_location("dendrogram_baseline", args.module_file)
        if spec is None or spec.loader is None:
            raise ValueError("Cannot load baseline module")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    n = args.classes
    distances = squareform(pdist(np.random.default_rng(42).normal(size=(n, 8))))
    linkage = module.linkage
    if args.deep:
        z = np.array([[0, 1, 1.0, 2]] + [[n + i - 1, i + 1, 1.0 + i / n, i + 2] for i in range(1, n - 1)])

        def linkage(*args, **kwargs):
            return z

    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    with patch.object(module, "linkage", linkage):
        fig, info = module._plot_probabilistic_dendrogram(
            distances, [f"Species {i}" for i in range(n)], apriori=[f"Group {i % 5}" for i in range(n)]
        )
    rendered = time.perf_counter()
    path = args.output / "dendrogram.svg"
    try:
        with rc_context({"svg.fonttype": "none"}):
            fig.savefig(path, bbox_inches="tight")
    finally:
        plt.close(fig)
    result = {
        "classes": n,
        "deep": args.deep,
        "baseline": str(args.module_file) if args.module_file else None,
        "render_seconds": rendered - started,
        "total_seconds": time.perf_counter() - started,
        "svg_bytes": path.stat().st_size,
        "svg_paths": path.read_text().count("<path"),
        "labels": len(info["label"]),
        "scope": "CPU linkage, layout, rendering and SVG export; excludes model distance computation and name lookup",
    }
    report = json.dumps(result, indent=2) + "\n"
    (args.output / "measurement.json").write_text(report)
    print(report, end="")


if __name__ == "__main__":
    main()
