import io
import math
import sys
from argparse import ArgumentParser
from contextlib import contextmanager
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

try:
    from Bio import Phylo
    from Bio.Phylo.BaseTree import BranchColor
    from pycirclize import Circos
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
            "Dendrogram visualization requires optional dependencies: biopython, pycirclize, scipy. "
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
        if not isinstance(cls2idx.get("0", None), dict):
            cls2idx = {0: cls2idx}
        idx2cls = {int(level): {v: k for k, v in c2i.items()} for level, c2i in cls2idx.items()}
    elif not isinstance(idx2cls.get("0", None), dict):
        idx2cls = {0: idx2cls}
    idx2cls = {int(k): v for k, v in idx2cls.items()}

    class_names = [[str(idx2cls[i].get(j, idx2cls[i].get(str(j), j))) for j in range(len(idx2cls[i]))] for i in range(len(idx2cls))]
    orig_class_names = [c.copy() for c in class_names]
    apriori: list[None | list[str] | dict[str, str]] = [None] * len(class_names)

    try:
        # Attempt to coerce to scientific names
        taxonomy = {cls: resolve_name_or_id(cls) for cls in class_names[0]}
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
            c2n = {cls: resolve_name_or_id(cls, rank_contains=None, skip=level)[TKC[level]][1] for cls in class_names[level]}
            n2c = {}
            for c, n in c2n.items():
                n2c.setdefault(n, []).append(c)
            c2c = {}
            for n, c in n2c.items():
                if len(c) > 1:
                    for ci in c:
                        c2c[ci] = resolve_name_or_id(ci, rank_contains=None, skip=level, full=True)[TKC[level]][1]
            # | End synonym resolution |
            if isinstance(ag, str):
                alevel = TKC.index(ag)
                apriori[level] = [c2p[c2n[cls]][alevel][1] for cls in class_names[level]]
            else:
                apriori[level] = ag
            class_names[level] = [c2c.get(cls, c2n[cls]) for cls in class_names[level]]
    except (RuntimeError, KeyError, ValueError):
        pass

    return [
        _plot_probabilistic_dendrogram(
            W=W, names=class_names[i], orig_names=orig_class_names[i], apriori=apriori[i], min_merge_prob=min_merge_prob, plot=plot
        )
        for i, W in enumerate(class_distance(model))
    ]


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
    condensed_dist = squareform(W, checks=False)
    Z = linkage(condensed_dist, method="ward")

    # Check if we actually have ground-truth colors to plot
    if not apriori or isinstance(apriori, str):
        apriori = {}
    elif isinstance(apriori, (list, tuple)):
        apriori = {cls: grp for cls, grp in zip(names, apriori)}
    apriori = {sanitize(k): v for k, v in apriori.items()}

    # --- 1. PROBABILISTIC CLUSTERING (EDGE COLORS) ---
    distance_threshold = -np.log(min_merge_prob) if min_merge_prob > 0 else 100
    clusters = fcluster(Z, t=distance_threshold, criterion="distance")

    cmap = plt.get_cmap("tab20")
    cluster_color_map = {cluster_id: mcolors.to_hex(cmap(i % 20)) for i, cluster_id in enumerate(sorted(set(clusters)))}
    apriori_color_map = {grp: mcolors.to_hex(cmap(i % 20)) for i, grp in enumerate(set(apriori.values()))}

    if plot:
        # --- 2. BUILD THE TREE ---
        newick_str = linkage_to_newick(Z, names)
        phylo_tree = Phylo.read(io.StringIO(newick_str), format="newick")  # pyright: ignore[reportPrivateImportUsage]

        # --- 3. DYNAMIC SCALING HEURISTICS ---
        num_classes = len(names)
        fig_size = min(40.0, max(10.0, num_classes / 50.0))
        radius_inches = fig_size * 0.4
        pts_per_label = (2 * math.pi * radius_inches * 72) / num_classes

        dynamic_font_size = min(12.0, max(0.5, pts_per_label * 0.8))
        dynamic_line_width = dynamic_font_size * 0.15

        # --- 4. INITIALIZE CIRCULAR DENDROGRAM ---
        rmargin = 5.0 if apriori else 0.5
        with temporary_recursion_limit(100000):
            circos, tv = Circos.initialize_from_tree(
                phylo_tree,
                r_lim=(30, 85),
                leaf_label_size=dynamic_font_size,
                leaf_label_rmargin=rmargin,
                line_kws=dict(lw=dynamic_line_width),
            )

        # --- 5. Apply colors ---
        for cluster, color in cluster_color_map.items():
            leaf_list = [names[i] for i, clst in enumerate(clusters) if clst == cluster]
            tv.set_node_line_props(leaf_list, color=color, apply_label_color=True)

        # --- 6. A PRIORI METADATA TRACK ---
        if apriori:
            sector = circos.sectors[0]
            color_track = sector.add_track((86, 89))

            for i, leaf in enumerate(phylo_tree.get_terminals()):
                grp = apriori.get(sanitize(leaf.name), None)
                if grp is not None:
                    x_start, x_end = i, i + 1
                    color = apriori_color_map[grp]
                    color_track.rect(x_start, x_end, r_lim=(86, 89), color=color, lw=0)

        # --- 7. EXPORT ---
        fig = circos.plotfig()
        fig.set_size_inches(fig_size, fig_size)
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
