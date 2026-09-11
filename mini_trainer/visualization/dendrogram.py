import sys
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from functools import lru_cache
from sqlite3 import Error as SQLiteError
from typing import cast, get_args

import matplotlib.colors as mcolors
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn

from mini_trainer import get_logger
from mini_trainer.config import Formatter
from mini_trainer.integrations import TK, resolve_name_or_id
from mini_trainer.modeling import class_distance, classification_module

from ._dendrogram_layout import render_linkage

try:
    from Bio.Phylo.BaseTree import BranchColor
    from scipy.cluster.hierarchy import ClusterNode, fcluster, linkage, to_tree
    from scipy.spatial.distance import squareform

    _HAS_DENDROGRAM_DEPS = True
except ImportError:
    _HAS_DENDROGRAM_DEPS = False


@contextmanager
def temporary_recursion_limit(new_limit: int):
    """Temporarily increases Python's recursion limit for a block of code."""
    old_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(new_limit, old_limit))
    try:
        yield
    finally:
        sys.setrecursionlimit(old_limit)


def _check_deps():
    if not _HAS_DENDROGRAM_DEPS:
        raise ImportError(
            "Dendrogram visualization requires optional dependencies: biopython and scipy. "
            "Install them with: `uv pip install mini_trainer[recommended]` or `uv sync --extra recommended`."
        )


def linkage_to_newick(Z: np.ndarray, labels: list[str] | tuple[str, ...]) -> str:
    """Safely converts Scipy Linkage to Newick, escaping reserved chars."""
    tree = cast(ClusterNode, to_tree(Z, False))
    assert isinstance(tree, ClusterNode)

    def escape_label(label: str) -> str:
        label_str = str(label)
        reserved = set("(),:;'[] \t\n")
        if not any(c in reserved for c in label_str):
            return label_str
        escaped_str = label_str.replace("'", "''")
        return f"'{escaped_str}'"

    escaped_labels = [escape_label(lbl) for lbl in labels]

    # Explicit stack for iterative post-order traversal (node, visited_flag)
    stack: list[tuple[ClusterNode, bool]] = [(tree, False)]
    node_str: dict[int, str] = {}

    while stack:
        node, visited = stack.pop()

        if node.is_leaf():
            node_str[node.id] = escaped_labels[node.id]
        elif visited:
            left_node = cast(ClusterNode, node.get_left())
            right_node = cast(ClusterNode, node.get_right())

            left_s = node_str.pop(left_node.id)
            right_s = node_str.pop(right_node.id)

            left_dist = (node.dist - left_node.dist) / 2
            right_dist = (node.dist - right_node.dist) / 2

            # Match original recursion child output structure
            node_str[node.id] = f"({right_s}:{right_dist},{left_s}:{left_dist})"
        else:
            # Post-order: push parent back (marked visited), then children
            stack.append((node, True))
            stack.append((node.get_right(), False))
            stack.append((node.get_left(), False))

    return f"{node_str[tree.id]};"


def hex_to_branchcolor(hex_str: str) -> "BranchColor":
    """Converts a standard hex color to BioPython's strict BranchColor object."""
    _check_deps()
    r, g, b = mcolors.to_rgb(hex_str)
    return BranchColor(int(r * 255), int(g * 255), int(b * 255))


def sanitize(x):
    x = str(x).strip().lower().strip("'").strip('"')
    x = " ".join(filter(bool, x.split(" ")))
    return x


@lru_cache(maxsize=16)
def _resolve_labels(labels: tuple[str, ...], level: int = 0, full: bool = False):
    """Batch cold lookups and reuse immutable class metadata across epochs."""
    get_logger().info(f"Resolving {len(labels):,} dendrogram labels at level {level}")
    try:

        def resolve(label):
            return resolve_name_or_id(label, rank_contains=None, skip=level, full=full)

        # Initialize the resolver's disk cache before concurrent requests.
        resolved = [resolve(labels[0])] if labels else []
        # Small bounded batches avoid queuing thousands of requests during an
        # outage; at most eight lookups need to finish before falling back.
        with ThreadPoolExecutor(max_workers=8) as executor:
            for start in range(1, len(labels), 8):
                resolved.extend(executor.map(resolve, labels[start : start + 8]))
        return dict(zip(labels, resolved, strict=True))
    except (RuntimeError, KeyError, ValueError, OSError, SQLiteError) as error:
        get_logger().info(f"Using original dendrogram labels: {error}")
        return None


def plot_probabilistic_dendrogram(
    model: nn.Module,
    min_merge_prob: float = 0.05,
    apriori_groups: list[list[str]] | list[dict[str, str]] | list[str] | dict[str, str] | str | bool | None = True,
    plot: bool = True,
):
    """Plot the probabilistic dendrogram for a model's class centers."""
    _check_deps()
    if apriori_groups is False:
        apriori_groups = None

    meta = classification_module(model).metadata
    idx2cls: dict = meta.get("idx2cls", {})
    cls2idx: dict = meta.get("cls2idx", {})
    if not idx2cls:
        if not isinstance(cls2idx.get("0", cls2idx.get(0)), dict):
            cls2idx = {0: cls2idx}
        idx2cls = {int(level): {v: k for k, v in c2i.items()} for level, c2i in cls2idx.items()}
    elif not (isinstance(idx2cls.get("0", idx2cls.get(0)), dict) and all(str(key).isdigit() for key in idx2cls.get("0", idx2cls.get(0)))):
        idx2cls = {0: idx2cls}
    idx2cls = {int(k): v for k, v in idx2cls.items()}

    class_names = [[str(idx2cls[i].get(j, idx2cls[i].get(str(j), j))) for j in range(len(idx2cls[i]))] for i in range(len(idx2cls))]
    orig_class_names = [c.copy() for c in class_names]
    apriori: list[None | list[str] | dict[str, str]] = [None] * len(class_names)

    try:
        # Attempt to coerce to scientific names
        taxonomy = _resolve_labels(tuple(class_names[0]), 0)
        if taxonomy is None:
            raise ValueError("Taxonomy unavailable")
        TKC = get_args(TK)
        get_logger().info("Class names successfully detected as species!")
        if apriori_groups is True:
            apriori_groups = list(TKC)[1:]
        if isinstance(apriori_groups, dict):
            apriori_groups = [apriori_groups]
        if isinstance(apriori_groups, str):
            apriori_groups = [apriori_groups]
        for level in range(len(class_names)):
            if apriori_groups and level < len(apriori_groups):
                ag = apriori_groups[level]
            else:
                ag = None
            c2p = {v[level][1]: v for v in (list(_v.values()) for _v in taxonomy.values())}
            # | Resolve untracked synonym conflicts |
            resolved = _resolve_labels(tuple(class_names[level]), level)
            if resolved is None:
                raise ValueError("Taxonomy unavailable")
            c2n = {cls: resolved[cls][TKC[level]][1] for cls in class_names[level]}
            n2c = {}
            for c, n in c2n.items():
                n2c.setdefault(n, []).append(c)
            c2c = {}
            for n, c in n2c.items():
                if len(c) > 1:
                    full_names = _resolve_labels(tuple(c), level, full=True)
                    if full_names is None:
                        continue
                    for ci in c:
                        c2c[ci] = full_names[ci][TKC[level]][1]
            # | End synonym resolution |
            if isinstance(ag, str):
                alevel = TKC.index(ag)
                apriori[level] = [c2p[c2n[cls]][alevel][1] for cls in class_names[level]]
            else:
                apriori[level] = ag
            class_names[level] = [c2c.get(cls, c2n[cls]) for cls in class_names[level]]
    except (RuntimeError, KeyError, ValueError):
        pass

    results = []
    try:
        for i, W in enumerate(class_distance(model)):
            get_logger().info(f"Rendering dendrogram level {i}: {len(class_names[i]):,} classes")
            results.append(
                _plot_probabilistic_dendrogram(
                    W=W,
                    names=class_names[i],
                    orig_names=orig_class_names[i],
                    apriori=apriori[i],
                    min_merge_prob=min_merge_prob,
                    plot=plot,
                )
            )
        return results
    except BaseException:
        for fig, _ in results:
            plt.close(fig)
        raise


def _plot_probabilistic_dendrogram(
    W: torch.Tensor | np.ndarray,
    names: list[str],
    orig_names: list[str] | None = None,
    apriori: dict[str, str] | list[str] | str | None = None,
    min_merge_prob: float = 0.05,
    plot: bool = True,
):
    if isinstance(W, torch.Tensor):
        W = W.numpy(force=True)
    if orig_names is None:
        orig_names = names.copy()
    if W.shape != (len(names), len(names)) or not names:
        raise ValueError("Dendrogram requires one label per row of a nonempty square distance matrix")
    condensed_dist = squareform(W, checks=False)
    if not np.isfinite(condensed_dist).all():
        raise ValueError("Dendrogram distances contain non-finite values")
    Z = linkage(condensed_dist, method="ward") if len(names) > 1 else np.empty((0, 4))

    # Check if we actually have ground-truth colors to plot
    if not apriori or isinstance(apriori, str):
        apriori = {}
    elif isinstance(apriori, (list, tuple)):
        apriori = {cls: grp for cls, grp in zip(names, apriori)}
    apriori = {sanitize(k): v for k, v in apriori.items()}

    # --- 1. PROBABILISTIC CLUSTERING (EDGE COLORS) ---
    distance_threshold = -np.log(min_merge_prob) if min_merge_prob > 0 else 100
    clusters = fcluster(Z, t=distance_threshold, criterion="distance") if len(names) > 1 else np.ones(1, dtype=int)

    cmap = plt.get_cmap("tab20")
    cluster_color_map = {cluster_id: mcolors.to_hex(cmap(i % 20)) for i, cluster_id in enumerate(sorted(set(clusters)))}
    apriori_color_map = {grp: mcolors.to_hex(cmap(i % 20)) for i, grp in enumerate(dict.fromkeys(apriori.values()))}

    if plot:
        fig = render_linkage(Z, names, clusters, cluster_color_map, apriori, apriori_color_map)
    else:
        fig = plt.figure()
    return fig, {
        "class": orig_names,
        "label": names,
        "apriori": [apriori.get(sanitize(cl), None) for cl in names],
        "cluster": [int(c) for c in clusters],
    }


def cli():
    description = (
        "Plot a dendrogram of the (leaf) classes in a model by visual similarity.\n"
        "Similarities are derived from the inner product between the parameters in the last layer"
        "under a uniform null distribution over the unit hypersphere.\n"
        "If the class names can be parsed as scientific species names or GBIF species IDs, then the "
        "dendrogram will automatically map the internal class names to the accepted species name via "
        "the GBIF API, and if no a priori groups are passed genera will be used as a priori groups (labels)."
    )
    parser = ArgumentParser(prog="plot_class_dendrogram", description=description, formatter_class=Formatter)
    parser.add_argument(
        "-w", "--weights", type=str, required=True, help="Model weights from which parameters and class names are extracted."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Path to save the dendrogram figure. OBS: It is highly recommended to specify an SVG output (i.e. ends with .svg).",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=None,
        required=False,
        dest="min_merge_prob",
        help="Class visual similarity (as fractional percent, e.g. 0.05) threshold for post-hoc clustering.",
    )
    parser.add_argument(
        "-l",
        "--labels",
        type=str,
        default="auto",
        required=False,
        help="Labels for a priori class clusters or groups. "
        "For example if the classes are species, then a priori class clusters or groups could be genera, family, or order etc."
        'Default is "auto" which attempts to use the subsequent level as labels e.g. genus for species, family for genus, etc.'
        'Use "no" for no labels.',
    )
    return parser.parse_args()


def run():
    from mini_trainer.builders import BaseBuilder

    kwargs = vars(cli())
    output: str = kwargs.pop("output")
    assert len(output) > 0

    apriori_groups: str | bool | None = kwargs.pop("labels", None)
    if isinstance(apriori_groups, str):
        match apriori_groups.strip().lower():
            case "auto":
                apriori_groups = True
            case "no":
                apriori_groups = False
            case _:
                pass

    model, _ = BaseBuilder.build_model(weights=kwargs.pop("weights"))
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    dendr = plot_probabilistic_dendrogram(model=model, apriori_groups=apriori_groups, **kwargs)
    for level, (fig, _) in enumerate(dendr):
        parts = output.split(".")
        path = ".".join(parts[: max(2, len(parts) - 1)])
        ext = "svg" if len(parts) <= 1 else parts[-1]
        if len(dendr) > 1:
            figname = f"{path}_{level}.{ext}"
        else:
            figname = f"{path}.{ext}"
        fig.savefig(figname, bbox_inches="tight")


if __name__ == "__main__":
    run()
