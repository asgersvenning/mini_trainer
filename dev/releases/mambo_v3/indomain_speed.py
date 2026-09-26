"""Render environment-specific throughput from the validated UCloud summary."""

import argparse
import csv
import hashlib
import json
import shutil
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dev.releases.mambo_v3.defaults_report import SERIES

parser = argparse.ArgumentParser(description="Render HPC request and streaming throughput separately")
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
root, out = args.source, args.output
out.mkdir(parents=True, exist_ok=True)
rows = list(csv.DictReader((root / "speed.csv").open()))
streams = list(csv.DictReader((root / "streaming_speed.csv").open()))
fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
plt.rcParams.update({"svg.fonttype": "none", "svg.hashsalt": "mambo-indomain-speed-v1"})
for ax, device in zip(axes[:2], ["cpu", "cuda:0"]):
    for model, label, color in SERIES:
        subset = [r for r in rows if r["device"] == device and r["preset"] == "full" and r["variant"] == model]
        batches = sorted({int(r["batch_size"]) for r in subset})
        groups = [[float(r["images_per_second"]) for r in subset if int(r["batch_size"]) == b] for b in batches]
        med = [statistics.median(g) for g in groups]
        ax.errorbar(
            batches,
            med,
            yerr=[[m - min(g) for m, g in zip(med, groups)], [max(g) - m for m, g in zip(med, groups)]],
            label=label,
            color=color,
            marker="o",
            capsize=3,
        )
    ax.set(
        xscale="log",
        xlabel="Batch size",
        ylabel="Images/s",
        title="EPYC CPU · request" if device == "cpu" else "B200 · request",
        xticks=batches,
    )
    ax.set_xticklabels(batches)
    ax.grid(alpha=0.15)
ax = axes[2]
for i, (model, label, color) in enumerate(SERIES[1:]):
    vals = [float(r["images_per_second"]) for r in streams if r["device"] == "cuda:0" and r["preset"] == "full" and r["variant"] == model]
    med = statistics.median(vals)
    ax.barh(i, med, color=color)
    ax.errorbar(med, i, xerr=[[med - min(vals)], [max(vals) - med]], color="black", capsize=3)
ax.set(yticks=range(4), yticklabels=[r[1] for r in SERIES[1:]], xlabel="Images/s", title="B200 · streaming, batch 256")
ax.invert_yaxis()
fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=5)
fig.text(
    0.02,
    0.02,
    "Global vocabulary; median and range of 3 process trials. CPU runtime threads: 4. Streaming: 1,024 images, startup included.\n"
    "Request and streaming are different execution modes; these are measured pipeline rates, not GPU throughput ceilings.",
    fontsize=10,
)
fig.tight_layout(rect=(0, 0.1, 1, 0.91))
fig.savefig(out / "mambo-indomain-speed.svg", metadata={"Date": None})
path = out / "mambo-indomain-speed.svg"
path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")

# Retain the exact source observations and campaign identities alongside the figure.
for source, destination in (("speed.csv", "speed.csv"), ("streaming_speed.csv", "streaming-speed.csv")):
    shutil.copy2(root / source, out / ("mambo-indomain-" + destination))
data = json.loads((root / "ucloud-summary.json").read_text())
provenance = {k: data[k] for k in ("environment_id", "fingerprint", "metric_revision", "policy", "sample_ids_sha256", "timing_bank_sha256")}
for name in ("ucloud-summary.json", "speed.csv", "streaming_speed.csv"):
    with (root / name).open("rb") as stream:
        provenance.setdefault("source_sha256", {})[name] = hashlib.file_digest(stream, "sha256").hexdigest()
(out / "mambo-indomain-campaign.json").write_text(json.dumps(provenance, indent=2) + "\n")
