"""Validation-selected PCA/Gaussian-NB dimension and chart operation timings."""

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
from benchmark import Coordinates, load, scores
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score
from sklearn.naive_bayes import GaussianNB
from threadpoolctl import threadpool_limits


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    full, f, _ = load(a.input)
    result = {"rows": [], "timings": {}, "cpu": platform.processor()}
    for seed in [42, 43, 44]:
        s = np.load(a.baseline / f"split-{seed}.npz")
        keep = s["eligible_original_rows"]
        x, y = full[keep], f[keep]
        tr, va, te = [s[k] for k in ["train", "validation", "test"]]
        pca = PCA(n_components=None, svd_solver="covariance_eigh").fit(x[tr])
        ztr, zva, zte = [pca.transform(x[idx]) for idx in [tr, va, te]]
        candidates = []
        best = None
        for d in [32, 128, 256, 512, 768, 1024, 1280]:
            clf = GaussianNB().fit(ztr[:, :d], y[tr])
            v = float(balanced_accuracy_score(y[va], clf.predict(zva[:, :d])))
            candidates.append({"dimensions": d, "validation_macro_recall": v})
            if best is None or v > best[0]:
                best = (v, d, clf)
        v, d, clf = best
        row = {"seed": seed, "selected_dimensions": d, "validation_candidates": candidates, "test": scores(y[te], clf.predict(zte[:, :d]))}
        result["rows"].append(row)
        if seed == 42:
            result["training_pca_variance"] = {
                str(d): float(pca.explained_variance_ratio_[:d].sum()) for d in [2, 32, 128, 256, 512, 768, 1024, 1280]
            }
            result["family_count"] = len(np.unique(y))
            result["train_validation_test_sizes"] = [len(tr), len(va), len(te)]
            result["training_mean_norm"] = float(np.linalg.norm(x[tr].mean(0)))
            for kind in ["ambient", "log_mean", "log_intrinsic", "stereographic", "equal_area", "log_radial"]:
                t = time.perf_counter()
                chart = Coordinates(kind).fit(x[tr])
                fit = time.perf_counter() - t
                q = x[te[:1]]
                z = chart.transform(q)
                item = {"fit_seconds": fit}
                for op, fn in [("insert", lambda: chart.transform(q)), ("inverse", lambda: chart.inverse(z))]:
                    times = []
                    for _ in range(101):
                        t = time.perf_counter()
                        fn()
                        times.append(time.perf_counter() - t)
                    item[op + "_median_us"] = float(np.median(times) * 1e6)
                if kind == "log_intrinsic":
                    c = np.clip(x[tr] @ chart.mu, -1, 1)
                    tail = x[tr] - c[:, None] * chart.mu
                    angle = np.arctan2(np.linalg.norm(tail, axis=1), c)
                    grad = (tail * (angle / np.maximum(np.linalg.norm(tail, axis=1), 1e-15))[:, None]).mean(0)
                    item["mean_log_norm_after_40_steps"] = float(np.linalg.norm(grad))
                result["timings"][kind] = item
        (a.output / "results.json").write_text(json.dumps(result, indent=2))
        print(row, flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        main()
