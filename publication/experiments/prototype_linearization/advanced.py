"""PCA/whitening, Nyström kernels and fast principal nested spheres follow-up.

PNS uses Monem, Dryden & George (2025), section 3.3, for the initial
32-dimensional sphere. Least-squares subspheres use three starts with L-BFGS.
This is a bounded reference implementation, not the authors' R package.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from benchmark import Coordinates, load, scores, unit
from scipy.optimize import minimize, minimize_scalar
from sklearn.decomposition import PCA
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from threadpoolctl import threadpool_limits


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class PNS:
    def fit(self, x):
        self.stages = []
        self.optimization = []
        scale = 1.0
        while x.shape[1] > 2:
            _, eig = np.linalg.eigh(np.cov(x, rowvar=False))
            starts = [unit(x.mean(0)), eig[:, 0], eig[:, 1]]

            def objective(v):
                nv = np.linalg.norm(v)
                u = v / nv
                c = np.clip(x @ u, -1 + 1e-12, 1 - 1e-12)
                angles = np.arccos(c)
                residual = angles - angles.mean()
                grad = x.T @ (-2 * residual / np.sqrt(1 - c * c)) / len(x)
                grad = (grad - u * np.dot(grad, u)) / nv
                return np.mean(residual**2), grad

            fits = [
                minimize(objective, v, method="L-BFGS-B", jac=True, options={"maxiter": 100, "ftol": 1e-11, "gtol": 1e-7}) for v in starts
            ]
            best = min(fits, key=lambda f: f.fun)
            axis = unit(best.x)
            radius = np.arccos(np.clip(x @ axis, -1, 1)).mean()
            if radius > np.pi / 2:
                axis = -axis
                radius = np.pi - radius
            w = axis.copy()
            w[0] -= 1
            w = unit(w) if np.linalg.norm(w) > 1e-12 else np.zeros_like(w)
            self.stages.append((w, radius, scale))
            self.optimization.append(
                {"dimension": x.shape[1] - 1, "success": bool(best.success), "iterations": int(best.nit), "objective": float(best.fun)}
            )
            rotated = x - 2 * (x @ w)[:, None] * w
            x = unit(rotated[:, 1:])
            scale *= np.sin(radius)
        angles = np.arctan2(x[:, 1], x[:, 0])
        # Bounded scalar minimization in several intervals to address the seam.
        candidates = [
            minimize_scalar(lambda m: np.mean(wrap(angles - m) ** 2), bounds=(left, left + np.pi / 2), method="bounded")
            for left in np.linspace(-np.pi, np.pi, 4, endpoint=False)
        ]
        self.circle_mean = min(candidates, key=lambda f: f.fun).x
        self.circle_scale = scale
        return self

    def transform(self, x):
        parts = []
        for w, radius, scale in self.stages:
            y = x - 2 * (x @ w)[:, None] * w
            parts.append((np.arctan2(np.linalg.norm(y[:, 1:], axis=1), y[:, 0]) - radius) * scale)
            x = unit(y[:, 1:])
        parts.append(wrap(np.arctan2(x[:, 1], x[:, 0]) - self.circle_mean) * self.circle_scale)
        return np.column_stack(parts[::-1])

    def inverse(self, z):
        angle = z[:, 0] / self.circle_scale + self.circle_mean
        x = np.column_stack((np.cos(angle), np.sin(angle)))
        for i, (w, radius, scale) in enumerate(self.stages[::-1], 1):
            angle = z[:, i] / scale + radius
            x = np.column_stack((np.cos(angle), np.sin(angle)[:, None] * x))
            x -= 2 * (x @ w)[:, None] * w
        return x


def evaluate(features, family, genus, split):
    tr, va, te = split
    ztr, zva, zte = features
    scale = np.sqrt(np.mean(np.sum((ztr - ztr.mean(0)) ** 2, axis=1)))
    ztr, zva, zte = [z / scale for z in features]
    best = None
    for alpha in [0.01, 0.1, 1, 10, 100]:
        clf = RidgeClassifier(alpha=alpha).fit(ztr, family[tr])
        v = balanced_accuracy_score(family[va], clf.predict(zva))
        if best is None or v > best[0]:
            best = (v, alpha, clf)
    row = {"ridge_family": scores(family[te], best[2].predict(zte)) | {"alpha": best[1], "validation_balanced_accuracy": float(best[0])}}
    g = GaussianNB().fit(ztr, family[tr])
    row["gaussian_nb_family"] = scores(family[te], g.predict(zte))
    covered = np.isin(genus[te], genus[tr])
    knn = KNeighborsClassifier(n_neighbors=5, weights="distance").fit(ztr, genus[tr])
    row["knn_genus"] = scores(genus[te][covered], knn.predict(zte)[covered])
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    full, f, g = load(a.input)
    results = []
    for seed in a.seeds:
        s = np.load(a.baseline / f"split-{seed}.npz")
        keep = s["eligible_original_rows"]
        x = full[keep]
        family = f[keep]
        genus = g[keep]
        split = [s[k] for k in ["train", "validation", "test"]]
        tr, va, te = split

        def record(name, features, fit_s, insert_s, inverse=None, extra=None):
            row = {
                "seed": seed,
                "method": name,
                "dimensions": features[0].shape[1],
                "fit_seconds": fit_s,
                "transform_seconds_all": insert_s,
            }
            row.update(evaluate(features, family, genus, split))
            if inverse:
                restored = unit(inverse(features[2]))
                row["reconstruction_degrees"] = float(np.degrees(np.arccos(np.clip(np.sum(restored * x[te], axis=1), -1, 1))).mean())
            if extra:
                row.update(extra)
            results.append(row)
            (a.output / "results.json").write_text(json.dumps(results, indent=2))
            print(seed, name, row["ridge_family"], row["gaussian_nb_family"], "fit", round(fit_s, 2), flush=True)

        start = time.perf_counter()
        pca = PCA(n_components=None, svd_solver="covariance_eigh").fit(x[tr])
        fit_s = time.perf_counter() - start
        start = time.perf_counter()
        pc = [pca.transform(x[idx]) for idx in split]
        insert_s = time.perf_counter() - start
        for dim in [32, 128, 1280]:
            for whiten in [False, True]:
                if dim == 1280 and not whiten:
                    continue
                scales = np.sqrt(np.maximum(pca.explained_variance_[:dim], 1e-12)) if whiten else np.ones(dim)
                features = [z[:, :dim] / scales for z in pc]
                record(
                    f"pca{dim}" + ("_whiten" if whiten else ""),
                    features,
                    fit_s,
                    insert_s,
                    lambda z: (z * scales) @ pca.components_[:dim] + pca.mean_,
                )
        # Fast PNS: PCA of orthogonal tangent coordinates, log projection,
        # exp onto S^32, then nested small-sphere least-squares fitting.
        start = time.perf_counter()
        chart = Coordinates("log_mean").fit(x[tr])
        rot = chart.rotate(x[tr])
        tangent = rot[:, 1:]
        smallpca = PCA(n_components=32, svd_solver="randomized", random_state=seed).fit(tangent)
        basis = smallpca.components_
        logs = [chart.transform(x[idx]) @ basis.T for idx in split]

        def sphere(z):
            r = np.linalg.norm(z, axis=1)
            return np.column_stack((np.cos(r), z * np.sinc(r[:, None] / np.pi)))

        spheres = [sphere(z) for z in logs]
        preparation = time.perf_counter() - start
        record("fast_pns_input_log32", logs, preparation, 0, lambda z: chart.inverse(z @ basis))
        start = time.perf_counter()
        pns = PNS().fit(spheres[0])
        fit_s = time.perf_counter() - start + preparation
        start = time.perf_counter()
        features = [pns.transform(z) for z in spheres]
        insert_s = time.perf_counter() - start
        error = float(np.max(np.abs(pns.inverse(features[2]) - spheres[2])))
        assert error < 1e-8, error

        def invert_pns(z):
            reduced = pns.inverse(z)
            return chart.rotate(np.column_stack((reduced[:, 0], reduced[:, 1:] @ basis)))

        record(
            "fast_pns32",
            features,
            fit_s,
            insert_s,
            invert_pns,
            {"reduced_sphere_roundtrip_max_abs": error, "optimizer_stages": pns.optimization},
        )
        # Kernel features for classical linear models; no analytic inverse.
        for gamma in [1.0, 5.0, 20.0]:
            start = time.perf_counter()
            kernel = Nystroem(kernel="rbf", gamma=gamma, n_components=512, random_state=seed).fit(x[tr])
            fit_s = time.perf_counter() - start
            start = time.perf_counter()
            features = [kernel.transform(x[idx]) for idx in split]
            insert_s = time.perf_counter() - start
            preimage = Ridge(alpha=0.01).fit(features[0], x[tr])
            record(
                f"nystrom512_gamma{gamma}",
                features,
                fit_s,
                insert_s,
                preimage.predict,
                {"inverse": "ridge preimage, fitted on training only; approximate"},
            )
    print("Complete", flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        main()
