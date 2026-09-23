"""Render reproducible pairwise overlap for the pinned release preset lists."""

import argparse
import csv
import tomllib
from pathlib import Path

from dev.releases.mambo_v3.audit import HERE, sha256

ASSETS = HERE.parents[2] / "docs/assets"


def overlap(left, right):
    shared = len(left & right)
    union = len(left | right)
    return shared, shared / union if union else 0.0, shared / len(left) if left else 0.0


def render(preview=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    manifest = tomllib.loads((HERE / "preset-manifest.toml").read_text())
    if sha256(HERE / "preset-definitions.toml") != manifest["definitions_sha256"]:
        raise ValueError("Rebuild presets before plotting modified definitions")
    names = list(manifest["presets"])
    sets = {}
    for name, item in manifest["presets"].items():
        path = HERE / item["path"]
        labels = path.read_text().splitlines()
        if sha256(path) != item["sha256"] or len(labels) != len(set(labels)) or len(labels) != item["count"]:
            raise ValueError(f"Preset integrity mismatch: {name}")
        sets[name] = set(labels)
    scores = [[overlap(sets[left], sets[right]) for right in names] for left in names]
    ASSETS.mkdir(exist_ok=True)
    with (ASSETS / "preset-overlap.tsv").open("w", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(["row_preset", "column_preset", "shared_species", "jaccard_percent", "row_coverage_percent"])
        for i, left in enumerate(names):
            for j, right in enumerate(names):
                shared, jaccard, coverage = scores[i][j]
                writer.writerow([left, right, shared, f"{jaccard * 100:.6f}", f"{coverage * 100:.6f}"])
    plt.rcParams.update({"svg.hashsalt": "mambo-preset-overlap-v1", "font.size": 9})
    fig, axes = plt.subplots(1, 2, figsize=(25, 13), layout="constrained")
    labels = [f"{name.replace('_', ' ')} ({len(sets[name]):,})" for name in names]
    for ax, metric, title in zip(axes, (1, 2), ("Jaccard: shared / union (%)", "Coverage: row species also in column (%)")):
        values = np.array([[cell[metric] * 100 for cell in row] for row in scores])
        im = ax.imshow(values, vmin=0, vmax=100, cmap="viridis")
        ax.set_xticks(range(len(names)), labels, rotation=60, ha="right", rotation_mode="anchor")
        ax.set_yticks(range(len(names)), labels)
        ax.set_title(title, fontsize=15, pad=18)
        for i in range(len(names)):
            for j in range(len(names)):
                value = values[i, j]
                label = "<1" if 0 < value < 1 else f"{value:.0f}"
                ax.text(j, i, label, ha="center", va="center", fontsize=7, color="black" if value > 55 else "white")
        fig.colorbar(im, ax=ax, shrink=0.65, pad=0.015)
    fig.suptitle(
        "MAMBO release presets — species overlap\n"
        "Qualified lists; full omitted. Regional scope and evidence thresholds differ. Coverage is directional.",
        fontsize=18,
    )
    fig.savefig(ASSETS / "preset-overlap.svg", metadata={"Date": None})
    svg = ASSETS / "preset-overlap.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    if preview:
        fig.savefig(preview, dpi=110)
    plt.close(fig)
    print(f"Wrote {ASSETS / 'preset-overlap.svg'} and pairwise TSV for {len(names)} presets")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", type=Path)
    args = parser.parse_args()
    render(args.preview)
