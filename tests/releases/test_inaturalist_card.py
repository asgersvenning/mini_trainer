"""The card comparison cannot substitute labels, hide missing taxa or publish smoke results."""

import hashlib
import json

import pytest

from dev.releases.mambo_v3 import inaturalist_card as comparison
from dev.releases.mambo_v3.package_download_metadata import card_performance


def test_cv_uses_visual_score_not_geography_or_response_order():
    first = {"vision_score": 20, "combined_score": 99, "taxon": {"id": 1}}
    second = {"vision_score": 80, "combined_score": 1, "taxon": {"id": 2}}
    assert comparison.cv_prediction({"results": [first, second]}) is second
    with pytest.raises(ValueError, match="visual scores"):
        comparison.cv_prediction({"results": [{"taxon": {"id": 1}}]})


@pytest.mark.parametrize("match,expected", [("EXACT", "123"), ("FUZZY", "inat:99"), ("NONE", "inat:99")])
def test_unresolved_taxonomy_is_retained_without_guessing(monkeypatch, tmp_path, match, expected):
    monkeypatch.setattr(
        comparison,
        "cached_json",
        lambda *a, **kw: {
            "matchType": match,
            "rank": "SPECIES",
            "order": "Lepidoptera",
            "usageKey": 456,
            "acceptedUsageKey": 123,
        },
    )
    assert comparison.gbif_label(None, {"id": 99, "name": "Example species"}, tmp_path) == expected


def test_above_species_prediction_is_not_given_a_descendant(monkeypatch, tmp_path):
    monkeypatch.setattr(comparison, "cached_json", lambda *a, **kw: {"results": [{"ancestors": [{"rank": "family"}]}]})
    assert comparison.species_taxon(None, {"id": 1, "rank": "genus"}, tmp_path) is None


def test_card_rejects_missing_smoke_and_tampered_evidence(tmp_path):
    with pytest.raises(ValueError, match="1,000"):
        card_performance(tmp_path, required=True)
    figures = {}
    for name in ("quality.png", "speed.png"):
        (tmp_path / name).write_bytes(b"reviewed figure")
        figures[name] = hashlib.sha256(b"reviewed figure").hexdigest()
    summary = {
        "count": 10,
        "species_count": 7,
        "unmapped_truth": 0,
        "models": dict.fromkeys(comparison.MODELS),
        "figures": figures,
        "cutoff": "2026-10-02T00:00:00Z",
    }
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="1,000"):
        card_performance(tmp_path, required=True)
    summary["count"] = 1000
    path.write_text(json.dumps(summary))
    assert "performance/quality.png" in card_performance(tmp_path, required=True)
    (tmp_path / "quality.png").write_bytes(b"changed figure")
    with pytest.raises(ValueError, match="Changed card figure"):
        card_performance(tmp_path, required=True)


def test_runtime_bootstrap_omits_binary_card_assets(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from dev.releases.mambo_v3 import package_download_metadata as packaging

    source = tmp_path / "source"
    source.mkdir()
    (source / "classes.json").write_text('{"labels": []}')
    manifest = {"files": {"classes.json": {}, "performance/quality.png": {}}, "origins": {}}
    monkeypatch.setattr(packaging, "Bundle", lambda root: SimpleNamespace(manifest=manifest, file=lambda name: source / name))
    target = tmp_path / "bootstrap.json"
    packaging.package(source, target)
    metadata = json.loads(target.read_text())["metadata"]
    assert json.loads(metadata["classes.json"]) == {"labels": []}
    assert "performance/quality.png" not in json.loads(metadata["release.json"])["files"]


def test_selection_reuses_only_frozen_prefix(tmp_path):
    from types import SimpleNamespace

    records = [{"id": i} for i in range(1500)]
    path = tmp_path / "samples.json"
    path.write_text(json.dumps({"records": records}))
    original = path.read_bytes()
    selected = comparison.selected_records(SimpleNamespace(output=tmp_path, count=1000))
    assert selected == records[:1000]
    assert path.read_bytes() == original


@pytest.mark.parametrize("changed", ["cpu", "platform", "versions", "identity"])
def test_resume_rejects_changed_runtime_before_reusing_predictions(tmp_path, changed):
    provenance = {"threads": 4}
    environment = {"cpu": "CPU A", "platform": "Linux", "versions": {"numpy": "2"}}
    comparison.start_runtime(tmp_path, provenance, environment)
    comparison.start_runtime(tmp_path, provenance, environment)
    previous = comparison.read(tmp_path / "runtime.json")
    previous[changed] = "changed"
    (tmp_path / "runtime.json").write_text(json.dumps(previous))
    with pytest.raises(ValueError, match="CPU or runtime"):
        comparison.start_runtime(tmp_path, provenance, environment)
    assert comparison.read(tmp_path / "runtime.json") == previous


def test_timing_report_requires_comparable_completed_local_runs():
    from copy import deepcopy

    first = {
        "status": "complete",
        "cpu": "CPU A",
        "platform": "Linux",
        "versions": {"numpy": "2", "onnxruntime": "1"},
        "identity": {"threads": 4},
    }
    second = deepcopy(first)
    second["versions"] = {"numpy": "2", "open-clip-torch": "3"}
    comparison.check_timing_environments([first, second])
    for key, value in [("cpu", "CPU B"), ("status", "running"), ("versions", {"numpy": "3"}), ("identity", {"threads": 8})]:
        changed = {**second, key: value}
        with pytest.raises(ValueError):
            comparison.check_timing_environments([first, changed])
