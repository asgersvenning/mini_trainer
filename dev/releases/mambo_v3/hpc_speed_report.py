"""Publish the completed four-variant B200 smoke without mixing historical campaigns."""

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

from dev.releases.mambo_v3.defaults_report import SERIES


def publish(source, output, baseline):
    output.mkdir(parents=True, exist_ok=True)
    provenance = {"environment": json.loads((source / "environment.json").read_text()), "reports": {}}
    observations = []
    baseline_rows = list(csv.DictReader(baseline.open()))
    provenance["retained_request_baseline"] = {
        "file": baseline.name,
        "sha256": hashlib.sha256(baseline.read_bytes()).hexdigest(),
        "scope": "CPU and smaller GPU batches; earlier campaign, not rerun",
    }
    plt.rcParams.update({"svg.fonttype": "none", "svg.hashsalt": "mambo-hpc-current-speed-v1"})
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    for ax, device in zip(axes[:2], ("cpu", "cuda:0")):
        all_batches = set()
        for variant, label, color in SERIES:
            rows = [
                r
                for r in baseline_rows
                if r["device"] == device
                and r["preset"] == "full"
                and r["variant"] == variant
                and not (device == "cuda:0" and variant != "v2" and int(r["batch_size"]) == 256)
            ]
            batches = sorted({int(r["batch_size"]) for r in rows})
            all_batches.update(batches)
            groups = [[float(r["images_per_second"]) for r in rows if int(r["batch_size"]) == b] for b in batches]
            medians = [statistics.median(g) for g in groups]
            ax.errorbar(
                batches,
                medians,
                yerr=[[m - min(g) for m, g in zip(medians, groups)], [max(g) - m for m, g in zip(medians, groups)]],
                color=color,
                label=label,
                marker="o",
                markerfacecolor="white",
                capsize=3,
            )
        if device != "cpu":
            all_batches.add(256)
        ticks = sorted(all_batches)
        ax.set(xscale="log", xlabel="Batch size", ylabel="Images/s", xticks=ticks)
        ax.set_xticklabels(ticks)
        ax.xaxis.set_minor_formatter(NullFormatter())
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
        for mode in ("end_to_end", "streaming"):
            data = cell[mode]
            images = cell["batch_size"] if mode == "end_to_end" else data["images"]
            rates = [images / seconds for seconds in data["seconds"]]
            rate = images / data["median_seconds"]
            spread = [[rate - min(rates)], [max(rates) - rate]]
            if mode == "end_to_end":
                axes[1].errorbar(256, rate, yerr=spread, color=color, marker="D", capsize=3, linestyle="none")
            else:
                axes[2].barh(row, rate, color=color, height=0.58)
                axes[2].errorbar(rate, row, xerr=spread, color="#333333", capsize=3)
                axes[2].text(max(rates) + 28, row, f"{rate:,.0f}", va="center", fontsize=10)
            observations.extend(
                dict(variant=variant, mode=mode, repetition=i + 1, images=images, seconds=seconds, images_per_second=images / seconds)
                for i, seconds in enumerate(data["seconds"])
            )
    for ax, title in zip(axes, ("EPYC CPU · request", "B200 · request", "B200 · streaming, batch 256")):
        ax.set_title(title)
        ax.grid(alpha=0.15)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[2].set(xlabel="Images/s", xlim=(0, 2400), yticks=range(4), yticklabels=[s[1] for s in SERIES[1:]])
    axes[2].invert_yaxis()
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=5)
    fig.text(
        0.02,
        0.025,
        "Global vocabulary · circles/lines: retained earlier CPU and GPU request measurements; diamonds/bars: updated B200 batch 256.\n"
        "Median and range; earlier: 3 process trials. Updated: 3 repetitions per variant; streaming: 4,096 warm images including startup.\n"
        "Different campaigns are not joined as a scaling curve. Updated runtime: "
        f"{provenance['environment']['commit'][:7]}. Settings and source observations: linked evidence.",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.15, 1, 0.91))
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
    parser.add_argument("--baseline", type=Path, default=Path("docs/assets/mambo-indomain-speed.csv"))
    args = parser.parse_args()
    publish(args.source, args.output, args.baseline)
