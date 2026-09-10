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


def test_short_arc_chords_respect_geometric_error_and_keep_endpoints():
    from matplotlib.path import Path

    from mini_trainer.visualization._dendrogram_layout import _arc, _point

    tolerance = 0.002
    for radius in (32.75, 82.25, 89):
        for sweep in np.linspace(-np.pi, np.pi, 101):
            vertices, codes = [], []
            _arc(vertices, codes, 0.4, 0.4 + sweep, radius, tolerance)
            np.testing.assert_allclose(vertices[-1], _point(0.4 + sweep, radius))
            if codes == [Path.LINETO]:
                # Maximum deviation from a circular arc is at its midpoint.
                midpoint = (np.array(_point(0.4, radius)) + vertices[-1]) / 2
                assert radius - np.linalg.norm(midpoint) <= tolerance + 1e-12
            else:
                assert all(code == Path.CURVE4 for code in codes)
                previous = np.array(_point(0.4, radius))
                t = np.linspace(0, 1, 101)[:, None]
                for index in range(0, len(vertices), 3):
                    p1, p2, end = np.asarray(vertices[index : index + 3])
                    curve = (1 - t) ** 3 * previous + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t**2 * p2 + t**3 * end
                    assert np.max(np.abs(np.linalg.norm(curve, axis=1) - radius)) <= 4.3e-6 * radius
                    previous = end
    vertices, codes = [], []
    _arc(vertices, codes, 0, 0.001, 89, tolerance)
    assert codes == [Path.LINETO]


def test_compact_svg_preserves_text_styles_transforms_and_rounds_only_paths(tmp_path):
    from mini_trainer.visualization._svg import compact_svg, save_dendrogram_svg

    fig, _ = d._plot_probabilistic_dendrogram(np.ones((3, 3)) - np.eye(3), ["A & B", "same", "same"])
    try:
        original = io.StringIO()
        with rc_context({"svg.fonttype": "none"}):
            fig.savefig(original, format="svg", bbox_inches="tight")
        source = original.getvalue()
        compact = compact_svg(source)
        before, after = ElementTree.fromstring(source), ElementTree.fromstring(compact)
        ns = {"s": "http://www.w3.org/2000/svg"}
        css = "".join(node.text or "" for node in after.findall(".//s:style", ns))
        for old, new in zip(before.findall(".//s:text", ns), after.findall(".//s:text", ns), strict=True):
            assert old.text == new.text
            assert f"text.{new.attrib['class']}{{{old.attrib['style']}}}" in css
            assert {k: v for k, v in old.attrib.items() if k != "style"} == {k: v for k, v in new.attrib.items() if k != "class"}
        assert compact_svg(compact) == compact
        path = tmp_path / "tree.svg"
        save_dendrogram_svg(fig, path)
        assert len(ElementTree.parse(path).findall(".//s:text", ns)) == 3
        fixture = '<svg><defs></defs><path d="M -0.00001 1.23456 L 1e2 -2.34567"/><text x="1.23456">1.23456</text></svg>'
        assert 'd="M 0 1.235 L 100 -2.346"' in compact_svg(fixture)
        assert '<text x="1.23456">1.23456</text>' in compact_svg(fixture)
    finally:
        plt.close(fig)
