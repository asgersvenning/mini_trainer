"""Offline scientific contracts: priors, class alignment and spherical diagnostics."""

import json

import numpy as np
import pandas as pd
import pytest

from publication.experiments.training_ablations.analysis import analyze, prediction_analysis, prototype_analysis
from publication.experiments.training_ablations.data import digest, write_json


def test_balanced_prior_and_errors_exclude_correct_predictions():
    target = np.array([0, 1, 1, 1, 2, 2])
    logits = np.full((6, 3), -20.0)
    logits[np.arange(6), target] = 20
    summary, classes, flows, reliability, pairs = prediction_analysis(logits, target, [1, 10, 100])
    np.testing.assert_allclose(classes.predicted_mass_balanced, 1 / 3)
    np.testing.assert_allclose(classes.predicted_mass_empirical, [1 / 6, 3 / 6, 2 / 6])
    assert summary["frequency_recall"]["spearman"] is None
    assert summary["balanced"]["accuracy"] == pytest.approx(1)
    assert flows.error_probability.sum() == 0
    assert pairs.empty
    assert reliability.groupby("prior").mass.sum().tolist() == pytest.approx([1, 1])


def test_balanced_scores_and_group_flows_have_explicit_denominators():
    target = np.array([0, 1, 1, 1, 2, 2])
    logits = np.tile([2.0, 0, 0], (6, 1))
    summary, _, flows, _, pairs = prediction_analysis(logits, target, [1, 10, 100])
    assert summary["empirical"]["accuracy"] == pytest.approx(1 / 6)
    assert summary["balanced"]["accuracy"] == pytest.approx(1 / 3)
    p = np.exp([2, 0, 0]) / np.exp([2, 0, 0]).sum()
    assert summary["balanced"]["nll"] == pytest.approx(-np.log(p).mean())
    assert summary["balanced"]["brier"] == pytest.approx((p * p).sum() - 2 / 3 + 1)
    row = flows[(flows.true_group == "head") & (flows.predicted_group == "tail")].iloc[0]
    assert row.error_probability == row.share_of_source_errors == 1
    assert pairs.errors.sum() == 5


def test_missing_class_is_explicit_and_invalid_targets_rejected():
    summary, classes, _, _, _ = prediction_analysis(np.eye(3)[:2], np.array([0, 1]), [1, 2, 3])
    assert summary["missing_classes"] == [2]
    assert np.isnan(classes.recall.iloc[2])
    with pytest.raises(ValueError, match="integer class"):
        prediction_analysis(np.eye(3), np.array([0, 1, 3]), [1, 2, 3])


def test_geometry_is_scale_invariant_and_reproducible():
    weights = np.eye(3)
    first, frame = prototype_analysis(weights, [1, 2, 3], samples=4096, seed=10)
    second, scaled = prototype_analysis(weights * np.array([1, 2, 3])[:, None], [1, 2, 3], samples=4096, seed=10)
    assert first == second
    np.testing.assert_allclose(frame.nearest_angle_degrees, 90)
    np.testing.assert_allclose(frame.null_win_probability, scaled.null_win_probability)
    np.testing.assert_allclose(frame.null_win_probability, 1 / 3, atol=0.03)
    assert first["effective_rank"] == pytest.approx(3)
    with pytest.raises(ValueError, match="tied prototypes"):
        prototype_analysis(np.ones((3, 4)), [1, 2, 3], samples=16)


def artifact_fixture(root):
    root.mkdir()
    frame = pd.DataFrame(
        {
            "label": [0, 1, 2],
            "speciesKey": ["a", "b", "c"],
            "genusKey": ["g", "g", "h"],
            "familyKey": ["f", "f", "f"],
            "split": ["validation"] * 3,
            "sample_id": ["a/x", "b/y", "c/z"],
        }
    )
    frame.to_parquet(root / "samples.parquet")
    write_json(root / "classes.json", {"cls2idx": {"a": 0, "b": 1, "c": 2}, "counts": [1, 2, 3]})
    write_json(root / "prepared.json", {"files": {n: digest(root / n) for n in ["classes.json", "samples.parquet"]}})
    attempt = root / "runs/full_seed42/attempt-000"
    attempt.mkdir(parents=True)
    write_json(
        attempt / "run.json", {"variant": "full", "seed": 42, "epochs": 10, "loss": "emla", "normalized": True, "regularization": True}
    )
    write_json(attempt / "evaluation.json", {"split": "validation"})
    np.savez(attempt / "predictions.npz", logits=np.eye(3), target=frame.label, sample_id=frame.sample_id.to_numpy(dtype=str))
    write_json(attempt / "complete.json", {n: digest(attempt / n) for n in ["run.json", "evaluation.json", "predictions.npz"]})
    return attempt


def test_artifact_analysis_checks_integrity_and_alignment(tmp_path):
    root = tmp_path / "study"
    attempt = artifact_fixture(root)
    reports = analyze(root, tmp_path / "analysis")
    assert reports[0]["metrics"]["balanced"]["accuracy"] == 1
    assert (tmp_path / "analysis/provenance.json").exists()
    with pytest.raises(FileExistsError):
        analyze(root, tmp_path / "analysis")
    np.savez(attempt / "predictions.npz", logits=np.eye(3), target=np.arange(3), sample_id=["b/y", "a/x", "c/z"])
    with pytest.raises(ValueError, match="changed artifact"):
        analyze(root, tmp_path / "corrupt")
    marker = json.loads((attempt / "complete.json").read_text())
    marker["predictions.npz"] = digest(attempt / "predictions.npz")
    write_json(attempt / "complete.json", marker)
    with pytest.raises(ValueError, match="sample identity"):
        analyze(root, tmp_path / "misaligned")


def test_taxonomic_error_denominators_and_parent_aggregation():
    from publication.experiments.training_ablations.analysis import parent_analysis, taxonomic_errors

    target = np.array([0, 0, 1, 2, 3, 4, 5])
    pred = np.array([1, 2, 1, 0, 5, 4, 5])
    logits = np.full((7, 6), -20.0)
    logits[np.arange(7), pred] = 20
    _, classes, _, _, pairs = prediction_analysis(logits, target, [1, 2, 3, 4, 5, 6])
    taxa = pd.DataFrame({"genusKey": ["a", "a", "b", "b", "c", "c"], "familyKey": ["x", "x", "y", "y", "z", "z"]})
    for column in taxa:
        labels = taxa[column].to_numpy()
        pairs["same_" + column] = labels[pairs.true_class] == labels[pairs.predicted_class]
    flows = taxonomic_errors(pairs, classes)
    total = flows[flows.true_group == "all"]
    assert total.error_probability.sum() == pytest.approx(3 / 6)
    assert total.share_of_source_errors.sum() == pytest.approx(1)
    reports, tables = parent_analysis(logits, target, [1, 2, 3, 4, 5, 6], taxa)
    assert reports["genus"]["empirical"]["accuracy"] == pytest.approx(4 / 7)
    assert tables["genus_classes"].descendant_species.tolist() == [2, 2, 2]


def test_embedding_sample_and_geometry_are_reproducible():
    from publication.experiments.training_ablations.embeddings import embedding_geometry, select_samples

    frame = pd.DataFrame({"label": [0, 0, 1, 1, 2, 2], "sample_id": list("abcdef")})
    assert select_samples(frame, 1).equals(select_samples(frame.sample(frac=1, random_state=2), 1))
    values = np.repeat(np.eye(3), 2, axis=0)
    result = embedding_geometry(values, frame.label.to_numpy(), [1, 2, 3])
    np.testing.assert_allclose(result.mean_within_class_angle, 0)
    np.testing.assert_allclose(result.nearest_centroid_angle, 90)


def test_epoch_dynamics_preserves_missing_support_and_artifact_provenance(tmp_path):
    from publication.experiments.training_ablations.dynamics import analyze_dynamics

    root = tmp_path / "study"
    attempt = artifact_fixture(root)
    run = json.loads((attempt / "run.json").read_text())
    run["id"] = "full_seed42"
    write_json(attempt / "run.json", run)
    manifest = json.loads((attempt / "complete.json").read_text())
    manifest["run.json"] = digest(attempt / "run.json")
    write_json(attempt / "complete.json", manifest)
    write_json(root / "plan.json", [run])
    path = attempt / "model/logs/figures/epoch-0001/Confusion_matrix_lvl0/counts.npz"
    path.parent.mkdir(parents=True)
    np.savez(path, rows=[0, 1], columns=[0, 0], counts=[1, 2], shape=[3, 3])
    analyze_dynamics(root, tmp_path / "curves")
    frame = pd.read_csv(tmp_path / "curves/epochs.csv")
    assert frame.accuracy.iloc[0] == pytest.approx(1 / 3)
    assert frame.macro_recall.iloc[0] == pytest.approx(1 / 2)
    assert np.isnan(frame.head_recall.iloc[0])
    provenance = json.loads((tmp_path / "curves/provenance.json").read_text())
    assert provenance["inputs"][run["id"]]["logs"][str(path.relative_to(attempt))] == digest(path)


def test_evidence_snapshot_keeps_planned_runs_and_reproduces_evaluation(tmp_path):
    from publication.experiments.evidence import export

    (tmp_path / "cohort").mkdir()
    study = tmp_path / "cohort/study"
    attempt = artifact_fixture(study)
    planned = [{**json.loads((attempt / "run.json").read_text()), "id": "full_seed42"}, {"variant": "ce", "seed": 42, "id": "ce_seed42"}]
    write_json(study / "plan.json", planned)
    write_json(study / "config.json", {"screening": True})
    write_json(attempt / "train.json", {"wall_seconds": 1.0})
    write_json(attempt / "evaluation.json", {"split": "validation", "accuracy": 1.0, "macro_recall": 1.0})
    marker = json.loads((attempt / "complete.json").read_text())
    write_json(attempt / "complete.json", {**marker, "evaluation.json": digest(attempt / "evaluation.json")})

    catalog = export([tmp_path / "cohort"], tmp_path / "snapshot")
    runs = pd.read_parquet(tmp_path / "snapshot/runs.parquet").set_index("run_id")
    assert runs.status.to_dict() == {"full_seed42": "complete", "ce_seed42": "not_started"}
    scores = pd.read_parquet(tmp_path / "snapshot" / catalog.loc[catalog.kind == "scores", "path"].item())
    np.testing.assert_array_equal(scores[["c0", "c1", "c2"]].to_numpy(), np.eye(3))
    manifest = json.loads((tmp_path / "snapshot/manifest.json").read_text())["files"]
    assert set(manifest) == set(catalog.path) | {"catalog.csv", "schemas.json", "README.md", "uv.lock", "pyproject.toml"}

    write_json(attempt / "evaluation.json", {"split": "validation", "accuracy": 0.5, "macro_recall": 1.0})
    write_json(attempt / "complete.json", {**marker, "evaluation.json": digest(attempt / "evaluation.json")})
    with pytest.raises(ValueError, match="do not reproduce"):
        export([tmp_path / "cohort"], tmp_path / "corrupt")
