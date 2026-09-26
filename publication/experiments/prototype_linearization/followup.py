"""Bounded follow-up: matched dimensions, broader RBF widths, PNS at 128D."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from advanced import PNS, evaluate
from benchmark import Coordinates, load, unit
from sklearn.decomposition import PCA
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    full, f, g = load(a.input)
    rows = []
    for seed in [42, 43, 44]:
        s = np.load(a.baseline / f"split-{seed}.npz")
        keep = s["eligible_original_rows"]
        x, family, genus = full[keep], f[keep], g[keep]
        split = [s[k] for k in ["train", "validation", "test"]]
        tr, va, te = split

        def run(name, fit, transform, inverse):
            t = time.perf_counter()
            obj = fit()
            fit_s = time.perf_counter() - t
            t = time.perf_counter()
            features = [transform(obj, x[idx]) for idx in split]
            transform_s = time.perf_counter() - t
            row = {"method": name, "seed": seed, "fit_seconds": fit_s, "transform_seconds_all": transform_s}
            row.update(evaluate(features, family, genus, split))
            restored = unit(inverse(obj, features[2]))
            row["reconstruction_degrees"] = float(np.degrees(np.arccos(np.clip(np.sum(restored * x[te], axis=1), -1, 1))).mean())
            if isinstance(obj, dict) and "pns" in obj:
                row["optimizer_stages"] = obj["pns"].optimization
                row["reduced_sphere_roundtrip_max_abs"] = float(np.max(np.abs(obj["pns"].inverse(features[2]) - obj["sphere"](x[te]))))
                assert row["reduced_sphere_roundtrip_max_abs"] < 1e-8
            timings = []
            for _ in range(31):
                t = time.perf_counter()
                transform(obj, x[te[:1]])
                timings.append(time.perf_counter() - t)
            row["one_point_median_us"] = float(np.median(timings) * 1e6)
            rows.append(row)
            (a.output / "results.json").write_text(json.dumps(rows, indent=2))
            print(seed, name, row["ridge_family"], row["gaussian_nb_family"], "fit", fit_s, flush=True)

        for dim in [512]:
            for whiten in [False, True]:
                run(
                    f"pca{dim}" + ("_whiten" if whiten else ""),
                    lambda: PCA(n_components=dim, whiten=whiten, svd_solver="randomized", random_state=seed).fit(x[tr]),
                    lambda m, z: m.transform(z),
                    lambda m, z: m.inverse_transform(z),
                )
        for gamma in [0.01, 0.1]:

            def fit_kernel():
                k = Nystroem(kernel="rbf", gamma=gamma, n_components=512, random_state=seed).fit(x[tr])
                back = Ridge(alpha=0.01).fit(k.transform(x[tr]), x[tr])
                return k, back

            run(f"nystrom512_gamma{gamma}", fit_kernel, lambda m, z: m[0].transform(z), lambda m, z: m[1].predict(z))

        def fit_pns():
            chart = Coordinates("log_mean").fit(x[tr])
            basis = PCA(n_components=128, svd_solver="randomized", random_state=seed).fit(chart.rotate(x[tr])[:, 1:]).components_

            def sphere(z):
                log = chart.transform(z) @ basis.T
                r = np.linalg.norm(log, axis=1)
                return np.column_stack((np.cos(r), log * np.sinc(r[:, None] / np.pi)))

            small = sphere(x[tr])
            pns = PNS().fit(small)
            return {"chart": chart, "basis": basis, "sphere": sphere, "pns": pns}

        def inverse_pns(m, z):
            small = m["pns"].inverse(z)
            return m["chart"].rotate(np.column_stack((small[:, 0], small[:, 1:] @ m["basis"])))

        run("fast_pns128", fit_pns, lambda m, z: m["pns"].transform(m["sphere"](z)), inverse_pns)
    print("Complete", flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        main()
