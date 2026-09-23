"""Collect the four benchmark stages and render a small comparison plot."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--study", type=Path, required=True)
a = p.parse_args()
base = json.loads((a.study / "baseline/results.json").read_text())
rows = base["rows"]
for name in ["advanced", "followup"]:
    rows += json.loads((a.study / name / "results.json").read_text())
selection = json.loads((a.study / "selection/results.json").read_text())
fields = {
    "ridge_macro": lambda r: r["ridge_family"]["balanced_accuracy"],
    "ridge_micro": lambda r: r["ridge_family"]["accuracy"],
    "gnb_macro": lambda r: r["gaussian_nb_family"]["balanced_accuracy"],
    "genus_knn_micro": lambda r: r["knn_genus"]["accuracy"],
}
summary = {}
for method in dict.fromkeys(r["method"] for r in rows):
    rr = [r for r in rows if r["method"] == method]
    assert len(rr) == 3, (method, len(rr))
    summary[method] = {
        name: {"mean": float(np.mean([fn(r) for r in rr])), "sd": float(np.std([fn(r) for r in rr], ddof=1))} for name, fn in fields.items()
    }
    summary[method]["fit_seconds_mean"] = float(np.mean([r["fit_seconds"] for r in rr]))
summary["validation_selected_pca_gnb"] = {
    "mean": float(np.mean([r["test"]["balanced_accuracy"] for r in selection["rows"]])),
    "sd": float(np.std([r["test"]["balanced_accuracy"] for r in selection["rows"]], ddof=1)),
    "dimensions": [r["selected_dimensions"] for r in selection["rows"]],
}
(a.study / "summary.json").write_text(json.dumps(summary, indent=2))
header = (
    "| Representation | Linear family macro recall | Linear family micro accuracy | "
    "Gaussian NB family macro recall | Genus 5-NN accuracy |\n|---|---:|---:|---:|---:|\n"
)
for name in [
    "ambient",
    "log_mean",
    "log_intrinsic",
    "stereographic",
    "equal_area",
    "log_radial",
    "pca128",
    "pca512",
    "pca1280_whiten",
    "fast_pns32",
    "fast_pns128",
    "nystrom512_gamma0.01",
]:
    r = summary[name]
    header += "| " + name + " | " + " | ".join(f"{100 * r[k]['mean']:.2f} ± {100 * r[k]['sd']:.2f}" for k in fields) + " |\n"
(a.study / "table.md").write_text(header)
methods = ["ambient", "log_mean", "stereographic", "pca128", "pca512", "pca1280_whiten", "fast_pns128", "nystrom512_gamma0.01"]
labels = [
    "Original weights",
    "Sphere log map",
    "Stereographic",
    "PCA 128",
    "PCA 512",
    "Full PCA whitening",
    "Fast PNS 128",
    "Nyström 512 (γ=.01)",
]
fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
for ax, key, title in zip(axes, ["ridge_macro", "gnb_macro"], ["Linear family classifier", "Gaussian naive Bayes family classifier"]):
    y = np.arange(len(methods))
    vals = [100 * summary[m][key]["mean"] for m in methods]
    err = [100 * summary[m][key]["sd"] for m in methods]
    ax.barh(y, vals, xerr=err, color="#237f8e", alpha=0.85)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Held-out macro recall (%)")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.2)
fig.suptitle(
    "Production prototypes: coordinate usefulness depends on the downstream model\n"
    "Three species splits; error bars show split SD, not confidence intervals",
    fontsize=12,
)
fig.savefig(a.study / "comparison.png", dpi=160)
fig.savefig(a.study / "comparison.svg")
plt.close(fig)
with (a.study / "SHA256SUMS").open("w") as f:
    for item in sorted(a.study.rglob("*")):
        if item.is_file() and item.name != "SHA256SUMS" and "__pycache__" not in str(item):
            f.write(hashlib.sha256(item.read_bytes()).hexdigest() + "  " + str(item.relative_to(a.study)) + "\n")
print(header)
