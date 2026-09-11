"""Offline prototype exploration using mini_trainer's existing diagnostic APIs."""

import argparse
import base64
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from matplotlib import colormaps
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from torch import nn

from mini_trainer.modeling import Classifier, class_distance, class_log_similarity, class_similarity
from mini_trainer.visualization._dendrogram_layout import linkage_layout

from .projection import angular_tsne, prototype_projections


def weight_model(weight: torch.Tensor) -> nn.Module:
    """A linear-only diagnostic head: values are already effective weights."""
    with torch.device("meta"):
        head = Classifier(weight.shape[1], weight.shape[0], hidden=False, normalized=False)
    head.linear = nn.Linear(weight.shape[1], weight.shape[0], bias=False, device="meta")
    head.linear.load_state_dict({"weight": weight}, assign=True)
    return nn.Sequential(head)


def load_prototypes(path: Path):
    """Read supported floating-point linear heads without building a backbone.

    This is deliberately not a general checkpoint loader. Reject other head
    families instead of treating a sublayer as the complete prototype space.
    """
    state = torch.load(path, map_location="cpu", weights_only=True)
    metas = [(key, value) for key, value in state.items() if key.endswith("._extra_state") and isinstance(value, dict)]
    if len(metas) != 1:
        raise ValueError("Expected exactly one classifier metadata record")
    key, meta = metas[0]
    if meta.get("classifier_class") not in {
        "mini_trainer.modeling.classifier:Classifier",
        "mini_trainer.hierarchical.model:HierarchicalClassifier",
    }:
        raise ValueError("Explorer supports only Classifier and HierarchicalClassifier linear prototypes")
    prefix = key.removesuffix("_extra_state") + "linear."
    linear = {key[len(prefix) :]: value for key, value in state.items() if key.startswith(prefix)}
    normalized = meta.get("normalized")
    weight_key = "parametrizations.weight.original1" if normalized else "weight"
    raw = linear.get(weight_key)
    if not isinstance(raw, torch.Tensor) or raw.ndim != 2 or raw.dtype != torch.float32:
        raise ValueError("Expected float32 linear weights; quantized/other checkpoint encodings are not supported")
    layer = nn.Linear(raw.shape[1], raw.shape[0], bias="bias" in linear)
    if normalized:
        layer = Classifier._normalize_layer(layer, orthogonal_init=False)
    layer.load_state_dict(linear, strict=True)
    weight = layer.weight.detach().clone()
    if not torch.isfinite(weight).all() or not (weight.norm(dim=1) > 0).all():
        raise ValueError("Prototype rows must be finite and nonzero")
    mapping = meta["cls2idx"]
    if all(isinstance(value, dict) for value in mapping.values()):
        mapping = mapping.get("0", mapping.get(0))
    if not mapping or sorted(mapping.values()) != list(range(len(weight))):
        raise ValueError("Class mapping must cover every prototype exactly once")
    names = [""] * len(weight)
    for name, index in mapping.items():
        names[index] = str(name)
    hierarchy = meta.get("labels", {})
    groups = [list(map(str, hierarchy.get(name, [name]))) for name in names]
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return (
        weight,
        names,
        groups,
        {
            "checkpoint_sha256": digest,
            "checkpoint": path.name,
            "classifier": meta["classifier_class"],
            "normalized": normalized,
            "norm_range": [weight.norm(dim=1).min().item(), weight.norm(dim=1).max().item()],
            "bias_range": [layer.bias.min().item(), layer.bias.max().item()] if layer.bias is not None else None,
            "analysis_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "torch_version": str(torch.__version__),
            "numpy_version": np.__version__,
        },
    )


def array_payload(array, dtype="<f4"):
    return base64.b64encode(np.asarray(array, dtype=dtype).tobytes()).decode("ascii")


def stable_scores(z):
    """Evaluate the same Gaussian quantities directly, without rounding a CDF.

    Retain the repository z-score; only its probability evaluation uses float64.
    Even float64 distances can underflow for extreme z, so retain log-tail too.
    """
    z = torch.as_tensor(z).double()
    return -torch.special.log_ndtr(z), torch.special.log_ndtr(-z) / math.log(10)


def block_extrema(matrix, order, size=640):
    """Equal blocks; exclude self-pairs and padding from both summaries."""
    block = max(1, math.ceil(len(order) / size))
    count = math.ceil(len(order) / block)
    lows, highs = np.empty((count, count)), np.empty((count, count))
    for row in range(count):
        ids = order[row * block : (row + 1) * block]
        values = matrix[ids][:, order].copy()
        values[np.arange(len(ids)), np.arange(row * block, row * block + len(ids))] = np.nan
        padded = np.pad(values, ((0, 0), (0, count * block - len(order))), constant_values=np.nan)
        blocks = padded.reshape(len(ids), count, block)
        # Singleton diagonal blocks have no off-diagonal pairs.
        valid = np.isfinite(blocks).any(axis=(0, 2))
        lows[row] = np.where(valid, np.where(np.isnan(blocks), np.inf, blocks).min(axis=(0, 2)), np.nan)
        highs[row] = np.where(valid, np.where(np.isnan(blocks), -np.inf, blocks).max(axis=(0, 2)), np.nan)
    return lows, highs, block


def analyze(weight, names, groups, *, neighbours=12, seed=42, metadata=None, include_tsne=False):
    n, dim = weight.shape
    if n < 2 or dim <= 2 or neighbours < 1:
        raise ValueError("Need >=2 classes, >2 dimensions and >=1 neighbour")
    model = weight_model(weight)
    print(f"Computing repository z-scores and distances for {n:,} × {dim:,}", flush=True)
    z = class_similarity(model, cdf=False)[0].numpy()
    distance = class_distance(model)[0].numpy()
    tail = 1 - class_similarity(model, cdf=True)[0].numpy()
    log_tail = class_log_similarity(model, complement=True)[0].numpy()
    assert np.isfinite(distance).all() and np.isfinite(z).all()
    k = min(neighbours, n - 1)
    # z is a separately labelled refinement of saturated distance ties.
    # Stable sorting uses checkpoint class order for exactly equal z values.
    ranks = [1, min(5, n - 1), min(12, n - 1), min(32, n - 1)]
    profile_neighbours = np.empty((n, len(ranks)), dtype=np.int32)
    near = np.empty((n, k), dtype=np.int32)
    for i in range(n):
        row = z[i].copy()
        row[i] = -np.inf
        ordered = np.argsort(-row, kind="stable")
        near[i] = ordered[:k]
        profile_neighbours[i] = ordered[np.asarray(ranks) - 1]
    ids = np.column_stack((np.arange(n), near))
    local_d = distance[ids[:, :, None], ids[:, None, :]]
    local_z = z[ids[:, :, None], ids[:, None, :]]
    stable_near, _ = stable_scores(z[np.arange(n)[:, None], near])
    local_logtail = log_tail[ids[:, :, None], ids[:, None, :]] / math.log(10)
    zero_count = (distance == 0).sum(axis=1) - 1
    recovered = 0
    for row in range(n):
        zero_ids = np.flatnonzero(distance[row] == 0)
        zero_ids = zero_ids[zero_ids > row]
        stable_zero, _ = stable_scores(z[row, zero_ids])
        recovered += int((stable_zero > 0).sum())
    upper = np.triu_indices(n, 1) if n < 1024 else None
    rng = np.random.default_rng(seed)
    if upper is None:
        i, j = rng.integers(n, size=(2, 200_000))
        i, j = i[i != j], j[i != j]
    else:
        i, j = upper
    sampled = z[i, j]
    robust_range = np.quantile(log_tail[i, j] / math.log(10), [0.001, 0.999]).tolist()
    bins = np.linspace(-6, max(8, float(sampled.max()) + 0.1), 100)
    hist, edges = np.histogram(sampled, bins)
    stats = {
        "classes": n,
        "dimensions": dim,
        "off_diagonal_zero_distance_pairs": int(zero_count.sum() // 2),
        "classes_with_zero_distance_neighbours": int((zero_count > 0).sum()),
        "zero_pairs_resolved_by_float64_logcdf": recovered,
        "finite_off_diagonal_log_tail_pairs": int(np.isfinite(log_tail).sum() // 2),
        "pair_sample_count": len(sampled),
        "seed": seed,
        "pair_sampling": "all unordered pairs" if upper is not None else "uniform ordered pairs with replacement, diagonal excluded",
        "pair_z_mean": float(sampled.mean()),
        "pair_z_std": float(sampled.std()),
        "pair_z_quantiles": dict(zip(["1%", "50%", "99%", "max_sampled"], np.quantile(sampled, [0.01, 0.5, 0.99, 1]).tolist())),
        "nearest_z_median": float(np.median(z[np.arange(n), near[:, 0]])),
        "kth_z_median": float(np.median(z[np.arange(n), near[:, -1]])),
        "k": k,
        "histogram_excluded": int(len(sampled) - hist.sum()),
    }
    print(json.dumps(stats), flush=True)
    print("Building unchanged Ward linkage on class_distance", flush=True)
    tree = linkage(squareform(distance, checks=False), method="ward")
    order, _, _, _ = linkage_layout(tree, np.ones(n, dtype=int))
    order = np.asarray(order)
    low, high, block = block_extrema(tail, order)
    log_low, log_high, _ = block_extrema(log_tail, order)
    log_floor = math.floor(float(np.nanmin(log_low)) / math.log(10)) * math.log(10)
    del distance, tail
    ranks = [1, min(5, n - 1), min(12, n - 1), min(32, n - 1)]
    profiles = np.empty((n, len(ranks)), dtype=np.float32)
    for i in range(n):
        row = z[i].copy()
        row[i] = -np.inf
        profiles[i] = -np.partition(-row, np.asarray(ranks) - 1)[np.asarray(ranks) - 1]
    print("Computing PCA projections and neighbourhood retention", flush=True)
    projections = prototype_projections(weight, near.tolist())
    if include_tsne:
        print("Fitting angular t-SNE", flush=True)
        projections["Angular t-SNE"] = angular_tsne(z, dim, near.tolist(), seed)
    return {
        "projections": projections,
        "metadata": metadata or {},
        "stats": stats,
        "names": names,
        "groups": groups,
        "neighbours": near.tolist(),
        "local_distance": array_payload(local_d),
        "local_z": array_payload(local_z),
        "local_log10_tail": array_payload(local_logtail),
        "stable_neighbour_distance": array_payload(stable_near.numpy(), dtype="<f8"),
        "profiles": profiles.tolist(),
        "profile_ranks": ranks,
        "profile_neighbours": profile_neighbours.tolist(),
        "zero_count": zero_count.tolist(),
        "tree": tree.tolist(),
        "order": order.tolist(),
        "matrix_log10_floor": log_floor / math.log(10),
        "matrix_shape": list(low.shape),
        "matrix_min_values": array_payload(low),
        "matrix_max_values": array_payload(high),
        "matrix_log_min_values": array_payload(log_low),
        "matrix_log_max_values": array_payload(log_high),
        "robust_log10_range": robust_range,
        "colour_lut": colormaps["magma"](np.linspace(0, 1, 256), bytes=True)[:, :3].tolist(),
        "matrix_block": block,
        "histogram": {"edges": edges.tolist(), "counts": hist.tolist()},
    }


def synthetic_cases(dim, seed=42):
    rng = torch.Generator().manual_seed(seed)
    random = torch.nn.functional.normalize(torch.randn(256, dim, generator=rng), dim=1)
    centres = torch.nn.functional.normalize(torch.randn(4, dim, generator=rng), dim=1)
    clustered = torch.nn.functional.normalize(centres.repeat_interleave(64, dim=0) + 0.7 * random, dim=1)
    # Identical, orthogonal, antipodal and a near duplicate in the real width.
    axes = torch.eye(3, dim)
    algebra = torch.stack([axes[0], axes[0], -axes[0], axes[1], axes[2], axes[0] + 0.01 * axes[1]])
    algebra = torch.nn.functional.normalize(algebra, dim=1)
    return {"Independent unit vectors": random, "Four planted groups": clustered, "Algebraic edge cases": algebra}


def create_report(weights: Path, output: Path, *, include_tsne=True, synthetic=False):
    """Create a portable report from a supported mini_trainer weight file."""
    weights, output = Path(weights), Path(output)
    print("Reading effective prototype weights", flush=True)
    weight, names, groups, provenance = load_prototypes(weights)
    cases = {"Production checkpoint": analyze(weight, names, groups, metadata=provenance, include_tsne=include_tsne)}
    for name, sample in (synthetic_cases(weight.shape[1]) if synthetic else {}).items():
        labels = [f"synthetic-{i}" for i in range(len(sample))]
        cases[name] = analyze(
            sample, labels, [[label] for label in labels], metadata={"seed": 42, "synthetic": True}, include_tsne=include_tsne
        )
    print("Writing report artifacts", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(cases, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    (output / "report-data.json").write_text(payload)
    render_report(output, payload)
    summary = {name: {"metadata": case["metadata"], "stats": case["stats"]} for name, case in cases.items()}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Report: {output / 'explorer.html'}", flush=True)

    return output / "explorer.html"


def render_report(output, payload):
    """Render current viewer assets over preserved numerical data."""
    template = Path(__file__).with_name("report.html").read_text()
    (output / "explorer.html").write_text(
        template.replace("__REPORT_DATA__", payload)
        .replace("__PHOTO_SCRIPT__", Path(__file__).with_name("photos.js").read_text())
        .replace("__PROJECTION_SCRIPT__", Path(__file__).with_name("projection.js").read_text())
        .replace("__THUMBNAIL_SCRIPT__", Path(__file__).with_name("thumbnails.js").read_text())
        .replace("__STATE_SCRIPT__", Path(__file__).with_name("state.js").read_text())
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--angular-tsne", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    create_report(args.weights, args.output, include_tsne=args.angular_tsne, synthetic=True)


if __name__ == "__main__":
    main()
