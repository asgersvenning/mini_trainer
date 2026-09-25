"""Publish the completed four-variant B200 smoke without mixing historical campaigns."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dev.releases.mambo_v3.defaults_report import SERIES


def publish(source, output):
    output.mkdir(parents=True, exist_ok=True)
    provenance = {"environment": json.loads((source / "environment.json").read_text()), "reports": {}}
    observations = []
    plt.rcParams.update({"svg.fonttype": "none", "svg.hashsalt": "mambo-hpc-current-speed-v1"})
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharex=True, sharey=True)
    for row, (variant, label, color) in enumerate(SERIES[1:]):
        path = source / variant / "report.json"
        report = json.loads(path.read_text())
        if report["status"] != "complete":
            raise ValueError(f"Incomplete report: {variant}")
        cell = report["cells"][0]
        if cell["batch_size"] != 256 or cell["preset"] != "full":
            raise ValueError("Expected global preset and batch 256")
        provenance["reports"][variant] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            **{k: report[k] for k in ("runtime", "settings", "bundle_sha256", "manifest_sha256", "effective_precision")},
            "list_sha256": cell["list_sha256"],
            "peak_host_gib": report["peak_rss_kib_linux"] / 1024**2,
        }
        for ax, mode in zip(axes, ("end_to_end", "streaming")):
            data = cell[mode]
            images = cell["batch_size"] if mode == "end_to_end" else data["images"]
            rates = [images / seconds for seconds in data["seconds"]]
            rate = images / data["median_seconds"]
            ax.barh(row, rate, color=color, height=0.58)
            ax.errorbar(rate, row, xerr=[[rate - min(rates)], [max(rates) - rate]], color="#333333", capsize=3)
            ax.text(max(rates) + 28, row, f"{rate:,.0f}", va="center", fontsize=10)
            observations.extend(
                dict(variant=variant, mode=mode, repetition=i + 1, images=images, seconds=seconds, images_per_second=images / seconds)
                for i, seconds in enumerate(data["seconds"])
            )
    for ax, title in zip(axes, ("Request · 256 images per call", "Streaming · 4,096 images per pass")):
        ax.set(title=title, xlabel="Images/s", xlim=(0, 2400), yticks=range(4), yticklabels=[s[1] for s in SERIES[1:]])
        ax.grid(axis="x", alpha=0.15)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].invert_yaxis()
    fig.suptitle("Full NVIDIA B200 · V3 deployment at batch 256", fontsize=15)
    fig.text(
        0.02,
        0.025,
        "Global vocabulary; warm inputs; median and range of three repetitions per variant in one process.\n"
        "4 runtime threads; streaming: 48 preparation workers, 128 readers. PyTorch FP16; ONNX TF32. "
        f"Commit {provenance['environment']['commit'][:7]}.",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.12, 1, 0.93))
    path = output / "mambo-hpc-current-speed.svg"
    fig.savefig(path, metadata={"Date": None})
    plt.close(fig)
    path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    with (output / "mambo-hpc-current-speed.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(observations[0]))
        writer.writeheader()
        writer.writerows(observations)
    (output / "mambo-hpc-current-provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    publish(args.source, args.output)
