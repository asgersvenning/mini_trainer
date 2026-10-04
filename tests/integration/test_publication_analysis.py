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
