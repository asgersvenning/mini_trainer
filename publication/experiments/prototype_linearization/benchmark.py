"""Held-out prototype coordinate benchmark; research only, no training changes."""

import argparse
import csv
import gzip
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import scipy
import sklearn
from scipy.special import ndtri
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, adjusted_rand_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from threadpoolctl import threadpool_limits


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-15)


class Coordinates:
    def __init__(self, kind):
        self.kind = kind

    def fit(self, x):
        self.mu = unit(x.mean(0))
        if self.kind == "log_intrinsic":
            for _ in range(40):
                c = np.clip(x @ self.mu, -1, 1)
                t = x - c[:, None] * self.mu
                theta = np.arctan2(np.linalg.norm(t, axis=1), c)
                delta = (unit(t) * theta[:, None]).mean(0)
                r = np.linalg.norm(delta)
                if r < 1e-8:
                    break
                self.mu = unit(np.cos(r) * self.mu + np.sinc(r / np.pi) * delta)
        v = self.mu.copy()
        v[0] -= 1
        self.w = unit(v) if np.linalg.norm(v) > 1e-12 else np.zeros_like(v)
        if self.kind == "log_radial":
            radii = np.linalg.norm(self.base(x), axis=1)
            self.radii = np.sort(radii)
            # Smooth empirical-quantile interpolation, NOT an exact empirical CDF.
            # Gaussian radial asymptotic approximation; deliberately a control.
            n, d = x.shape
            self.targets = np.sqrt(d - 1 - 0.5) + ndtri((np.arange(n) + 0.5) / n) / np.sqrt(2)
        return self

    def rotate(self, x):
        return x - 2 * (x @ self.w)[..., None] * self.w

    def base(self, x):
        y = self.rotate(x)
        c, tail = np.clip(y[:, 0], -1, 1), y[:, 1:]
        r = np.linalg.norm(tail, axis=1)
        if self.kind == "stereographic":
            return tail / (1 + c[:, None])
        if self.kind == "equal_area":
            return tail * np.sqrt(2 / (1 + c[:, None]))
        return unit(tail) * np.arctan2(r, c)[:, None]

    def transform(self, x):
        if self.kind == "ambient":
            return x.copy()
        z = self.base(x)
        if self.kind == "log_radial":
            z = unit(z) * np.interp(np.linalg.norm(z, axis=1), self.radii, self.targets)[:, None]
        return z

    def inverse(self, z):
        if self.kind == "ambient":
            return unit(z)
        if self.kind == "log_radial":
            z = unit(z) * np.interp(np.linalg.norm(z, axis=1), self.targets, self.radii)[:, None]
        r = np.linalg.norm(z, axis=1)
        if self.kind == "stereographic":
            s = r * r
            return self.rotate(np.column_stack(((1 - s) / (1 + s), 2 * z / (1 + s[:, None]))))
        if self.kind == "equal_area":
            # Outside valid chart ball, clip for approximate PCA reconstruction.
            r = np.minimum(r, 2)
            z = unit(z) * r[:, None]
            return self.rotate(np.column_stack((1 - r * r / 2, z * np.sqrt(np.maximum(0, 1 - r * r / 4))[:, None])))
        return self.rotate(np.column_stack((np.cos(r), z * np.sinc(r[:, None] / np.pi))))


def scores(y, pred):
    return {"accuracy": float(accuracy_score(y, pred)), "balanced_accuracy": float(balanced_accuracy_score(y, pred))}


def load(path):
    with gzip.open(path, "rt", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    cols = [c for c in reader.fieldnames if c.startswith("weight_")]
    x = np.array([[r[c] for c in cols] for r in rows], dtype=np.float64)
    return unit(x), np.array([r["family_gbif_id"] for r in rows]), np.array([r["genus_gbif_id"] for r in rows])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    x, family, genus = load(a.input)
    _, counts = np.unique(family, return_counts=True)
    labels, counts = np.unique(family, return_counts=True)
    eligible = np.isin(family, labels[counts >= 10])
    x, family, genus = x[eligible], family[eligible], genus[eligible]
    result = {
        "input_sha256": hashlib.sha256(a.input.read_bytes()).hexdigest(),
        "shape": list(x.shape),
        "excluded_rare_family_rows": int((~eligible).sum()),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "platform": platform.platform(),
        "threads": 4,
        "rows": [],
    }
    for seed in a.seeds:
        tr, rest = train_test_split(np.arange(len(x)), test_size=0.3, random_state=seed, stratify=family)
        va, te = train_test_split(rest, test_size=0.5, random_state=seed, stratify=family[rest])
        np.savez(a.output / f"split-{seed}.npz", train=tr, validation=va, test=te, eligible_original_rows=np.flatnonzero(eligible))
        original_nn = NearestNeighbors(n_neighbors=10).fit(x[tr]).kneighbors(x[te], return_distance=False)
        for kind in ["ambient", "log_mean", "log_intrinsic", "stereographic", "equal_area", "log_radial"]:
            start = time.perf_counter()
            model = Coordinates(kind).fit(x[tr])
            fit_s = time.perf_counter() - start
            start = time.perf_counter()
            ztr, zva, zte = [model.transform(x[idx]) for idx in [tr, va, te]]
            transform_s = time.perf_counter() - start
            row = {
                "seed": seed,
                "method": kind,
                "fit_seconds": fit_s,
                "transform_seconds_all": transform_s,
                "heldout_roundtrip_max_abs": float(np.max(np.abs(model.inverse(zte) - x[te]))),
            }
            norms = np.linalg.norm(zte, axis=1)
            row["radius_mean_sd"] = [float(norms.mean()), float(norms.std())]
            # One scalar scaling lets the same regularization grid cover all charts.
            scale = np.sqrt(np.mean(np.sum((ztr - ztr.mean(0)) ** 2, axis=1)))
            features = [z / scale for z in [ztr, zva, zte]]
            best = None
            for alpha in [0.01, 0.1, 1, 10, 100]:
                clf = RidgeClassifier(alpha=alpha).fit(features[0], family[tr])
                val = balanced_accuracy_score(family[va], clf.predict(features[1]))
                if best is None or val > best[0]:
                    best = (val, alpha, clf)
            row["ridge_family"] = scores(family[te], best[2].predict(features[2])) | {"alpha": best[1]}
            gnb = GaussianNB().fit(features[0], family[tr])
            row["gaussian_nb_family"] = scores(family[te], gnb.predict(features[2]))
            nn = NearestNeighbors(n_neighbors=10).fit(ztr).kneighbors(zte, return_distance=False)
            row["original_neighbor_recall10"] = float(np.mean([len(set(a) & set(b)) / 10 for a, b in zip(nn, original_nn)]))
            for level, labels in [("family", family), ("genus", genus)]:
                covered = np.isin(labels[te], labels[tr])
                clf = KNeighborsClassifier(n_neighbors=5, weights="distance").fit(ztr, labels[tr])
                pred = clf.predict(zte)
                row["knn_" + level] = scores(labels[te][covered], pred[covered]) | {"test_coverage": float(covered.mean())}
            pca = PCA(n_components=128, svd_solver="randomized", random_state=seed).fit(ztr)
            row["pca"] = {}
            for dim in [2, 32, 128]:
                ptr, pte = pca.transform(ztr)[:, :dim], pca.transform(zte)[:, :dim]
                restored = model.inverse(pte @ pca.components_[:dim] + pca.mean_)
                angles = np.degrees(np.arccos(np.clip(np.sum(restored * x[te], axis=1), -1, 1)))
                row["pca"][str(dim)] = {
                    "mean_reconstruction_degrees": float(angles.mean()),
                    "test_coordinate_variance_explained": float(
                        1 - np.sum((zte - (pte @ pca.components_[:dim] + pca.mean_)) ** 2) / np.sum((zte - ztr.mean(0)) ** 2)
                    ),
                }
                if dim == 32:
                    km = MiniBatchKMeans(n_clusters=len(np.unique(family)), n_init=3, random_state=seed, batch_size=1024).fit(ptr)
                    row["pca"]["32"]["kmeans_family_ari"] = float(adjusted_rand_score(family[te], km.predict(pte)))
            row["total_seconds"] = time.perf_counter() - start + fit_s
            result["rows"].append(row)
            (a.output / "results.json").write_text(json.dumps(result, indent=2))
            print(
                seed,
                kind,
                "ridge",
                row["ridge_family"],
                "knn",
                row["knn_genus"]["accuracy"],
                "roundtrip",
                row["heldout_roundtrip_max_abs"],
                flush=True,
            )
    print("Complete", a.output, flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        main()
