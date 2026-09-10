"""Large-tree plotting regressions; no network or model downloads."""

import io
import sys
from unittest.mock import MagicMock
from xml.etree import ElementTree

import numpy as np
import pytest
from matplotlib import pyplot as plt
from matplotlib import rc_context

import mini_trainer.visualization.dendrogram as d
from mini_trainer.visualization._dendrogram_layout import linkage_layout

pytestmark = pytest.mark.skipif(not d._HAS_DENDROGRAM_DEPS, reason="Optional dendrogram dependencies missing")


@pytest.fixture(autouse=True)
def offline_labels(monkeypatch):
    d._resolve_labels.cache_clear()
    monkeypatch.setattr(d, "resolve_name_or_id", MagicMock(side_effect=ValueError("offline")))
    yield
    d._resolve_labels.cache_clear()


def test_iterative_layout_matches_existing_newick_geometry():
    from Bio import Phylo

    z = d.linkage(np.random.default_rng(42).normal(size=(32, 5)), method="ward")
    clusters = d.fcluster(z, 3, criterion="distance")
    order, angles, radii, groups = linkage_layout(z, clusters)
    tree = Phylo.read(io.StringIO(d.linkage_to_newick(z, [str(i) for i in range(32)])), "newick")
    assert order == [int(leaf.name) for leaf in tree.get_terminals()]
    depths = tree.depths()
    for leaf in tree.get_terminals():
        np.testing.assert_allclose(radii[int(leaf.name)], 32.75 + 49.5 * depths[leaf] / max(depths.values()), rtol=1e-6)
    for node, (left, right, height, _) in enumerate(z, 32):
        left, right = int(left), int(right)
        assert angles[node] == (angles[left] + angles[right]) / 2
        np.testing.assert_allclose(radii[node], 82.25 - 49.5 * height / z[-1, 2])
        assert groups[node] == (groups[left] if groups[left] == groups[right] else 0)


def test_deep_tree_exports_compact_svg_without_recursion_or_missing_labels(monkeypatch):
    n = 1100
    z = np.array([[0, 1, 1.0, 2]] + [[n + i - 1, i + 1, 1.0 + i / n, i + 2] for i in range(1, n - 1)])
    monkeypatch.setattr(d, "linkage", lambda *args, **kwargs: z)
    before = sys.getrecursionlimit()
    names = [f"Species {i}" for i in range(n)]
    fig, info = d._plot_probabilistic_dendrogram(np.zeros((n, n)), names, apriori=["family"] * n)
    try:
        assert info["class"] == names
        assert info["label"] == names
        assert info["cluster"] == d.fcluster(z, -np.log(0.05), criterion="distance").tolist()
        svg = io.StringIO()
        with rc_context({"svg.fonttype": "none"}):
            fig.savefig(svg, format="svg", bbox_inches="tight")
        root = ElementTree.fromstring(svg.getvalue())
        ns = {"s": "http://www.w3.org/2000/svg"}
        assert len(root.findall(".//s:text", ns)) == n
        assert len(root.findall(".//s:path", ns)) < 10
        assert sys.getrecursionlimit() == before
    finally:
        plt.close(fig)


@pytest.mark.parametrize("names", [["only"], ["same", "same", "other"]])
def test_singleton_and_duplicate_display_labels(names):
    w = np.ones((len(names), len(names))) - np.eye(len(names))
    fig, info = d._plot_probabilistic_dendrogram(w, names)
    try:
        assert info["label"] == names
        assert len(fig.axes[0].texts) == len(names)
        fig.canvas.draw()
    finally:
        plt.close(fig)


def test_name_resolution_is_bounded_and_failure_cached(monkeypatch):
    calls = []

    def fail(label, **kwargs):
        calls.append(label)
        raise OSError("offline")

    monkeypatch.setattr(d, "resolve_name_or_id", fail)
    labels = tuple(map(str, range(100)))
    assert d._resolve_labels(labels) is None
    count = len(calls)
    assert 1 <= count <= 8
    assert d._resolve_labels(labels) is None
    assert len(calls) == count


def test_taxonomy_cache_preserves_names_groups_and_refreshes_for_new_classes(monkeypatch):
    def resolve(label, **kwargs):
        return {"species": (label, f"Species {label}"), "genus": ("g", "Genus"), "family": ("f", "Family")}

    resolver = MagicMock(side_effect=resolve)
    monkeypatch.setattr(d, "resolve_name_or_id", resolver)
    metadata = {"idx2cls": {0: {0: "1", 1: "2"}}}
    monkeypatch.setattr(d, "classification_module", lambda _: MagicMock(metadata=metadata))
    monkeypatch.setattr(d, "class_distance", lambda _: [np.array([[0.0, 1.0], [1.0, 0.0]])])
    for _ in range(2):
        fig, info = d.plot_probabilistic_dendrogram(None)[0]
        plt.close(fig)
        assert info["class"] == ["1", "2"]
        assert info["label"] == ["Species 1", "Species 2"]
        assert info["apriori"] == ["Genus", "Genus"]
    assert resolver.call_count == 2
    metadata["idx2cls"][0][1] = "3"
    fig, info = d.plot_probabilistic_dendrogram(None)[0]
    plt.close(fig)
    assert info["label"] == ["Species 1", "Species 3"]


def test_failed_later_level_closes_earlier_figures(monkeypatch):
    metadata = {"idx2cls": {"0": {0: "a", 1: "b"}, "1": {0: "c", 1: "d"}}}
    monkeypatch.setattr(d, "classification_module", lambda _: MagicMock(metadata=metadata))
    monkeypatch.setattr(d, "class_distance", lambda _: [np.eye(2), np.full((2, 2), np.nan)])
    before = plt.get_fignums()
    with pytest.raises(ValueError, match="non-finite"):
        d.plot_probabilistic_dendrogram(None)
    assert plt.get_fignums() == before
