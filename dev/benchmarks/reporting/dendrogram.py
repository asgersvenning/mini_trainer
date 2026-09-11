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
from mini_trainer.visualization import save_dendrogram_svg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classes", type=int, default=3422)
    parser.add_argument("--deep", action="store_true")
    baseline = parser.add_mutually_exclusive_group()
    baseline.add_argument("--module-file", type=Path, help="Optional baseline dendrogram.py saved from Git")
    baseline.add_argument("--layout-file", type=Path, help="Optional baseline _dendrogram_layout.py saved from Git")
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
    renderer = module.render_linkage if hasattr(module, "render_linkage") else None
    if args.layout_file:
        spec = importlib.util.spec_from_file_location("layout_baseline", args.layout_file)
        layout = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(layout)
        renderer = layout.render_linkage
    n = args.classes
    distances = squareform(pdist(np.random.default_rng(42).normal(size=(n, 8))))
    linkage = module.linkage
    if args.deep:
        z = np.array([[0, 1, 1.0, 2]] + [[n + i - 1, i + 1, 1.0 + i / n, i + 2] for i in range(1, n - 1)])

        def linkage(*args, **kwargs):
            return z

    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    with patch.object(module, "linkage", linkage), patch.object(module, "render_linkage", renderer, create=True):
        fig, info = module._plot_probabilistic_dendrogram(
            distances, [f"Species {i}" for i in range(n)], apriori=[f"Group {i % 5}" for i in range(n)]
        )
    rendered = time.perf_counter()
    path = args.output / "dendrogram.svg"
    try:
        if args.module_file or args.layout_file:
            with rc_context({"svg.fonttype": "none"}):
                fig.savefig(path, bbox_inches="tight")
        else:
            save_dendrogram_svg(fig, path)
    finally:
        plt.close(fig)
    result = {
        "classes": n,
        "deep": args.deep,
        "baseline": str(args.module_file or args.layout_file) if args.module_file or args.layout_file else None,
        "path_vertices": sum(len(p.get_path().vertices) for ax in fig.axes for p in ax.patches),
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
