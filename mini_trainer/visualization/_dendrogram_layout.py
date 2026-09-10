"""Iterative circular linkage layout with SVG paths batched by branch color."""

import math
from collections import defaultdict

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import PathPatch, Wedge
from matplotlib.path import Path


def linkage_layout(linkage, clusters):
    """Return leaf order, angles, normalized radii and subtree cluster IDs.

    The right-child-first order matches the existing Newick exporter. Coordinates
    depend on node indices, not display labels (which may legitimately repeat).
    """
    n = len(clusters)
    children = linkage[:, :2].astype(int)
    order = []
    stack = [2 * n - 2]
    while stack:
        node = stack.pop()
        if node < n:
            order.append(node)
        else:
            left, right = children[node - n]
            stack.extend((left, right))
    angles = np.zeros(2 * n - 1)
    angles[order] = (np.arange(n) + 0.5) * 2 * np.pi / n
    groups = np.zeros(2 * n - 1, dtype=int)
    groups[:n] = clusters
    heights = np.zeros(2 * n - 1)
    heights[n:] = linkage[:, 2]
    for node, (left, right) in enumerate(children, n):
        angles[node] = (angles[left] + angles[right]) / 2
        if groups[left] == groups[right]:
            groups[node] = groups[left]
    if len(linkage) and heights[-1] == 0:
        # All-zero distances still need a visible tree; use topological heights.
        for node, (left, right) in enumerate(children, n):
            heights[node] = max(heights[left], heights[right]) + 1
    radii = 82.25 - 49.5 * heights / (heights[-1] or 1)
    return order, angles, radii, groups


def _point(angle, radius):
    return radius * math.sin(angle), radius * math.cos(angle)


def _arc(vertices, codes, start, end, radius):
    """Append circular cubic Bezier segments of at most 90 degrees."""
    steps = max(1, math.ceil(abs(end - start) / (math.pi / 2)))
    points = np.linspace(start, end, steps + 1)
    for a, b in zip(points[:-1], points[1:], strict=True):
        k = 4 / 3 * math.tan((b - a) / 4)
        x0, y0 = _point(a, radius)
        x1, y1 = _point(b, radius)
        vertices.extend(
            (
                (x0 + k * radius * math.cos(a), y0 - k * radius * math.sin(a)),
                (x1 - k * radius * math.cos(b), y1 + k * radius * math.sin(b)),
                (x1, y1),
            )
        )
        codes.extend((Path.CURVE4,) * 3)


def render_linkage(linkage, names, clusters, cluster_colors, apriori, apriori_colors):
    """Render every leaf and branch without recursive tree objects or searches."""
    n = len(names)
    order, angles, radii, groups = linkage_layout(linkage, clusters)
    fig_size = min(40.0, max(10.0, n / 50.0))
    font_size = min(12.0, max(0.5, 2 * math.pi * fig_size * 0.4 * 72 / n * 0.8))
    paths = defaultdict(lambda: ([], []))
    for parent, children in enumerate(linkage[:, :2].astype(int), n):
        left, right = children
        left_color = cluster_colors.get(groups[left], "black")
        right_color = cluster_colors.get(groups[right], "black")
        if left_color == right_color:
            # One continuous U-shaped branch replaces two radial/arc paths.
            vertices, codes = paths[left_color]
            vertices.extend((_point(angles[left], radii[left]), _point(angles[left], radii[parent])))
            codes.extend((Path.MOVETO, Path.LINETO))
            _arc(vertices, codes, angles[left], angles[right], radii[parent])
            vertices.append(_point(angles[right], radii[right]))
            codes.append(Path.LINETO)
            continue
        for child in children:
            vertices, codes = paths[cluster_colors.get(groups[child], "black")]
            vertices.extend((_point(angles[child], radii[child]), _point(angles[child], radii[parent])))
            codes.extend((Path.MOVETO, Path.LINETO))
            _arc(vertices, codes, angles[child], angles[parent], radii[parent])
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    try:
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_xlim(-105, 105)
        ax.set_ylim(-105, 105)
        for color, (vertices, codes) in paths.items():
            ax.add_patch(PathPatch(Path(vertices, codes), facecolor="none", edgecolor=color, lw=font_size * 0.15))

        # Merge adjacent taxonomy bands, then batch the remaining wedges by color.
        bands = defaultdict(list)
        labels = [apriori.get(" ".join(str(names[i]).strip().lower().strip("'").strip('"').split()), None) for i in order]
        start = 0
        while start < n:
            end = start + 1
            while end < n and labels[end] == labels[start]:
                end += 1
            if labels[start] is not None:
                wedge = Wedge((0, 0), 89, 90 - 360 * end / n, 90 - 360 * start / n, width=3)
                bands[apriori_colors[labels[start]]].append(wedge.get_path())
            start = end
        for color, wedges in bands.items():
            ax.add_patch(PathPatch(Path.make_compound_path(*wedges), facecolor=color, edgecolor="none"))
        label_radius = 87.25 if not apriori else 92.25
        for node in order:
            angle = 90 - math.degrees(angles[node])
            right = math.cos(math.radians(angle)) >= 0
            ax.text(
                *_point(angles[node], label_radius),
                names[node],
                fontsize=font_size,
                rotation=angle if right else angle + 180,
                rotation_mode="anchor",
                ha="left" if right else "right",
                va="center",
                color=cluster_colors[clusters[node]],
            )
        return fig
    except BaseException:
        plt.close(fig)
        raise
