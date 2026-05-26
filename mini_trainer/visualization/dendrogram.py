import io
import math

import matplotlib.colors as mcolors
import numpy as np
from matplotlib import pyplot as plt
from torch import nn

from mini_trainer.integrations import resolve_name_or_id
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


def _check_deps():
    if not _HAS_DENDROGRAM_DEPS:
        raise ImportError(
            "Dendrogram visualization requires optional dependencies: biopython, pycirclize, scipy. "
            "Install them with: `uv pip install mini_trainer[recommended]` or `uv sync --extra recommended`."
        )


def linkage_to_newick(Z: np.ndarray, labels: list[str] | tuple[str]) -> str:
    """Safely converts Scipy Linkage to Newick, escaping reserved chars."""
    tree = to_tree(Z, False)
    assert isinstance(tree, ClusterNode)

    def build_newick(node, newick, parentdist, leaf_names):
        if node.is_leaf():
            return f"{leaf_names[node.id]}:{(parentdist - node.dist) / 2}{newick}"
        else:
            if len(newick) > 0:
                newick = f"):{(parentdist - node.dist) / 2}{newick}"
            else:
                newick = ");"
            newick = build_newick(node.get_left(), newick, node.dist, leaf_names)
            newick = build_newick(node.get_right(), f",{newick}", node.dist, leaf_names)
            return f"({newick}"

    def escape_label(label: str) -> str:
        label_str = str(label)
        reserved = set("(),:;'[] \t\n")
        if not any(c in reserved for c in label_str):
            return label_str
        escaped_str = label_str.replace("'", "''")
        return f"'{escaped_str}'"

    return build_newick(tree, "", tree.dist, list(map(escape_label, labels)))


def hex_to_branchcolor(hex_str: str) -> "BranchColor":
    """Converts a standard hex color to BioPython's strict BranchColor object."""
    _check_deps()
    r, g, b = mcolors.to_rgb(hex_str)
    return BranchColor(int(r * 255), int(g * 255), int(b * 255))


def sanitize(x):
    x = str(x).strip().lower().strip("'").strip('"')
    x = " ".join(filter(bool, x.split(" ")))
    return x


def plot_probabilistic_dendrogram(model: nn.Module, min_merge_prob: float = 0.05, apriori_groups: list[str] | dict[str, str] | None = None):
    """Plot the probabilistic dendrogram for a model's class centers."""
    _check_deps()

    meta = classification_module(model).metadata
    idx2cls = meta.get("idx2cls", {})
    if not idx2cls:
        cls2idx = meta.get("cls2idx", {})
        if isinstance(cls2idx.get("0", None), dict):
            cls2idx = cls2idx["0"]
        idx2cls = {v: k for k, v in cls2idx.items()}
    if isinstance(idx2cls.get("0", None), dict):
        idx2cls = idx2cls["0"]

    class_names = [str(idx2cls.get(i, idx2cls.get(str(i), i))) for i in range(len(idx2cls))]
    try:
        # Attempt to coerce to scientific names
        class_names = [res["species"][1] for res in resolve_name_or_id(class_names)]
    except Exception:
        pass

    W = class_distance(model).cpu().numpy()
    condensed_dist = squareform(W)
    Z = linkage(condensed_dist, method="ward")

    # Check if we actually have ground-truth colors to plot
    if not apriori_groups:
        apriori_groups = {}
    elif isinstance(apriori_groups, (list, tuple)):
        apriori_groups = {cls: grp for cls, grp in zip(class_names, apriori_groups)}
    apriori_groups = {sanitize(k): v for k, v in apriori_groups.items()}

    # --- 1. PROBABILISTIC CLUSTERING (EDGE COLORS) ---
    distance_threshold = -np.log(min_merge_prob) if min_merge_prob > 0 else 100
    clusters = fcluster(Z, t=distance_threshold, criterion="distance")

    cmap = plt.get_cmap("tab20")
    cluster_color_map = {cluster_id: mcolors.to_hex(cmap(i % 20)) for i, cluster_id in enumerate(sorted(set(clusters)))}
    apriori_color_map = {grp: mcolors.to_hex(cmap(i % 20)) for i, grp in enumerate(set(apriori_groups.values()))}

    # --- 2. BUILD THE TREE ---
    newick_str = linkage_to_newick(Z, class_names)
    phylo_tree = Phylo.read(io.StringIO(newick_str), format="newick")

    # --- 3. DYNAMIC SCALING HEURISTICS ---
    num_classes = len(class_names)
    fig_size = min(40.0, max(10.0, num_classes / 50.0))
    radius_inches = fig_size * 0.4
    pts_per_label = (2 * math.pi * radius_inches * 72) / num_classes

    dynamic_font_size = min(12.0, max(0.5, pts_per_label * 0.8))
    dynamic_line_width = dynamic_font_size * 0.15

    # --- 4. INITIALIZE CIRCULAR DENDROGRAM ---
    rmargin = 5.0 if apriori_groups else 0.5

    circos, tv = Circos.initialize_from_tree(
        phylo_tree, r_lim=(30, 85), leaf_label_size=dynamic_font_size, leaf_label_rmargin=rmargin, line_kws=dict(lw=dynamic_line_width)
    )

    # --- 5. Apply colors ---
    for cluster, color in cluster_color_map.items():
        leaf_list = [class_names[i] for i, clst in enumerate(clusters) if clst == cluster]
        tv.set_node_line_props(leaf_list, color=color, apply_label_color=True)

    # --- 6. A PRIORI METADATA TRACK ---
    if apriori_groups:
        sector = circos.sectors[0]
        color_track = sector.add_track((86, 89))

        for i, leaf in enumerate(phylo_tree.get_terminals()):
            grp = apriori_groups.get(sanitize(leaf.name), None)
            if grp is not None:
                x_start, x_end = i, i + 1
                color = apriori_color_map[grp]
                color_track.rect(x_start, x_end, r_lim=(86, 89), color=color, lw=0)

    # --- 7. EXPORT ---
    fig = circos.plotfig()
    fig.set_size_inches(fig_size, fig_size)
    return fig
