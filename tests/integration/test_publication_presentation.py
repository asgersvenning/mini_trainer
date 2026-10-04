"""Presentation must preserve paired seeds and expose incomplete evidence."""

import json

import pytest

from publication.experiments.training_ablations.analysis import analyze
from publication.experiments.training_ablations.data import write_json
from publication.experiments.training_ablations.presentation import assemble, paired_contrasts
from tests.integration.test_publication_analysis import artifact_fixture


def test_contrasts_never_pair_across_seeds():
    plan = [{"variant": variant, "seed": seed} for variant in ("full", "ce") for seed in (42, 43)]
    rows = [{"variant": "full", "seed": 42, "balanced.accuracy": 0.8}, {"variant": "ce", "seed": 43, "balanced.accuracy": 0.5}]
    pairs, missing = paired_contrasts(plan, rows)
    assert not pairs
    assert {row["seed"] for row in missing} == {42, 43}
    rows.append({"variant": "ce", "seed": 42, "balanced.accuracy": 0.6})
    pairs, missing = paired_contrasts(plan, rows)
    assert pairs[0]["difference"] == pytest.approx(0.2)
    assert pairs[0]["seed"] == 42
    assert [row["seed"] for row in missing] == [43]


def test_assembly_exposes_missing_runs_and_rejects_wrong_or_duplicate_reports(tmp_path):
    study, report = tmp_path / "study", tmp_path / "report"
    artifact_fixture(study)
    analyze(study, report)
    run = json.loads((report / "report.json").read_text())["runs"][0]["run"]
    # Older run manifests omit id; production plans and reports retain it.
    run["id"] = "full_seed42"
    content = json.loads((report / "report.json").read_text())
    content["runs"][0]["run"] = run
    write_json(report / "report.json", content)
    write_json(study / "plan.json", [run, {**run, "id": "full_seed43", "seed": 43}])
    assemble(study, [report], tmp_path / "out", plots=False)
    coverage = json.loads((tmp_path / "out/coverage.json").read_text())
    assert not coverage["complete"]
    assert coverage["missing_runs"] == ["full_seed43"]
    with pytest.raises(ValueError, match="Missing 1 planned"):
        assemble(study, [report], tmp_path / "strict", require_complete=True, plots=False)
    assert not (tmp_path / "strict").exists()
    with pytest.raises(ValueError, match="Duplicate report"):
        assemble(study, [report, report], tmp_path / "duplicate", plots=False)
    write_json(study / "plan.json", [run])
    assemble(study, [report], tmp_path / "complete", require_complete=True, plots=False)
    assert json.loads((tmp_path / "complete/coverage.json").read_text())["complete"]
    write_json(study / "plan.json", [{**run, "seed": 99}])
    with pytest.raises(ValueError, match="differs from frozen plan"):
        assemble(study, [report], tmp_path / "changed-plan", plots=False)
    write_json(study / "prepared.json", {"different": "cohort"})
    with pytest.raises(ValueError, match="different prepared study"):
        assemble(study, [report], tmp_path / "wrong", plots=False)
