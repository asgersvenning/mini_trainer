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
        "models": {name: {"threshold": 0.5, "coverage": 0.8} for name in comparison.MODELS},
        "report_count": 900,
        "calibration_count": 100,
        "figures": figures,
        "cutoff": "2026-10-02T00:00:00Z",
    }
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="1,000"):
        card_performance(tmp_path, required=True)
    summary["count"] = 1000
    summary["models"]["nemo"].pop("threshold")
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="calibrated"):
        card_performance(tmp_path, required=True)
    summary["models"]["nemo"]["threshold"] = 0.5
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


def test_fetch_can_shrink_incomplete_collection_without_new_requests(tmp_path, monkeypatch):
    from types import SimpleNamespace

    spec = {"count": 5000, "cutoff": "2026-10-02T09:30:00Z"}
    (tmp_path / "selection.json").write_text(json.dumps(spec))
    images = tmp_path / "images"
    images.mkdir()
    responses = tmp_path / "responses"
    responses.mkdir()
    observations = []
    for i in range(3):
        (images / f"{i}.jpg").write_bytes(b"cached image")
        observations.append(
            {
                "id": i,
                "created_at": spec["cutoff"],
                "photos": [{"id": i, "url": "https://example/square.jpg"}],
                "taxon": {"id": i, "name": str(i), "rank": "species"},
            }
        )
        (responses / f"gbif-{i}.json").write_text(
            json.dumps({"matchType": "EXACT", "rank": "SPECIES", "order": "Lepidoptera", "usageKey": i})
        )
    (responses / "observations-1.json").write_text(json.dumps({"results": observations}))
    monkeypatch.setattr(comparison, "session", lambda: None)  # Any new HTTP request would fail.
    comparison.fetch(SimpleNamespace(output=tmp_path, count=2))
    manifest = comparison.read(tmp_path / "samples.json")
    assert manifest["count"] == 2 and manifest["cutoff"] == spec["cutoff"]
    assert [r["id"] for r in manifest["records"]] == [0, 1]
    assert comparison.read(tmp_path / "selection.json") == spec


def test_confidence_recovery_requires_real_scores():
    assert comparison.prediction_confidence({"confidence": 0.73}, "nemo") == 0.73
    assert comparison.prediction_confidence({"response": {"results": [{"vision_score": 73}]}}, "inaturalist") == 0.73
    with pytest.raises(ValueError, match="cache lacks confidence"):
        comparison.prediction_confidence({"label": "1"}, "meghan")
    with pytest.raises(ValueError, match="Invalid"):
        comparison.prediction_confidence({"confidence": float("nan")}, "nemo")


def test_report_uses_builtin_selection_and_preserves_inputs(tmp_path, monkeypatch):
    from types import SimpleNamespace

    pytest.importorskip("mini_metrics")
    from mini_metrics.data import MetricDF
    from mini_metrics.metrics import MacroF1, evaluate_file

    try:
        comparison.require_pinned_metrics()
    except ValueError:
        pytest.skip("requires the campaign mini_metrics revision")

    source = tmp_path / "predictions"
    source.mkdir()
    records = [{"id": i, "path": f"{i}.jpg", "label": str(i % 3)} for i in range(180)]
    comparison.write_json(source / "samples.json", {"records": records, "cutoff": "2026-10-02"})
    runtime = {"status": "complete", "cpu": "test", "platform": "test", "versions": {}, "identity": {"threads": 4}}
    for model in comparison.MODELS:
        folder = source / model
        folder.mkdir()
        if model != "inaturalist":
            comparison.write_json(folder / "runtime.json", runtime)
        for r in records:
            wrong = r["id"] % 4 == 0
            confidence = 0.1 if wrong else 0.9
            prediction = {"label": str((int(r["label"]) + 1) % 3) if wrong else r["label"], "seconds": 0.1, "known": True}
            if model == "inaturalist":
                prediction["response"] = {"results": [{"vision_score": confidence * 100}]}
            else:
                prediction["confidence"] = confidence
            comparison.write_json(folder / f"{r['id']}.json", prediction)
    before = {p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    monkeypatch.setattr(
        comparison, "charts", lambda folder, result: [(folder / n).write_bytes(b"figure") for n in ("quality.png", "speed.png")]
    )
    destination = tmp_path / "report"
    args = SimpleNamespace(output=source, report_output=destination, count=180)
    comparison.summarize(args)
    result = comparison.read(destination / "summary.json")
    assert set(result["reporting_ids"]).isdisjoint(result["calibration_ids"])
    assert result["report_count"] + result["calibration_count"] == 180
    for model in comparison.MODELS:
        expected = evaluate_file(
            MetricDF.from_source(destination / f"{model}.csv"),
            optimal=True,
            seed=42,
            opt_crit=MacroF1,
            eps=0.01,
            use_quantiles=True,
            simple=True,
            hierarchical=False,
            pattern=r"^(accuracy|f1|coverage|optimal_confidence_threshold)$",
            verbose=0,
        )
        actual = result["models"][model]
        assert actual["threshold"] == expected["optimal_confidence_threshold"][0]
        assert actual["threshold"] > 0.1
        assert actual["macro_f1"] == expected["f1"][0]
        assert actual["coverage"] == expected["coverage"][0]
    assert before == {p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    args.report_output = source / "preview"
    with pytest.raises(ValueError, match="outside"):
        comparison.summarize(args)
