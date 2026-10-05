"""Compound bootstrap contracts: disjoint parts, observation units, repeats and replicate 0."""

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

pytest.importorskip("mini_metrics")

from mini_metrics.metrics import evaluate_file  # noqa: E402

from publication.experiments.statistics import replicates  # noqa: E402


def images(n=60, per_observation=2):
    species = [f"s{i % 6}" for i in range(n)]
    return pd.DataFrame(
        {
            "image_id": [f"img{i}" for i in range(n)],
            "observation_id": [f"obs{i // per_observation}" for i in range(n)],
            "speciesKey": species,
            "genusKey": [f"g{int(s[1:]) % 3}" for s in species],
            "familyKey": "f0",
        }
    )


def test_partitions_are_disjoint_and_resample_whole_observations():
    frame = images()
    weights = replicates.draw_replicates(frame, replicates=20, seed=1, calibration_fraction=0.2)
    assert weights.shape == (60, 21)
    assert set(np.unique(weights[:, 0])) == {-1, 1}  # Replicate 0: partition only.
    observations = frame.observation_id.to_numpy()
    calibration_units = frame.observation_id[weights[:, 0] < 0].nunique()
    assert calibration_units == 6  # 20% of each species' observations.
    for b in range(weights.shape[1]):
        # Images of one observation share their draw count and part.
        assert (pd.Series(weights[:, b]).groupby(observations).nunique() == 1).all()
        unit = pd.Series(weights[:, b]).groupby(observations).first()
        # Each part draws as many units as it holds: 30 in total, 6 for calibration.
        assert unit.abs().sum() == 30 and -unit[unit < 0].sum() == calibration_units
    np.testing.assert_array_equal(weights, replicates.draw_replicates(frame, 20, 1, 0.2))


def test_replicate_zero_matches_mini_metrics_and_repeats_count(tmp_path):
    rng = np.random.default_rng(0)
    n = 40
    labels = np.array([[f"s{i % 4}", f"g{i % 2}", "f0"] for i in range(n)])
    predictions = np.where(rng.uniform(size=(n, 3)) < 0.7, labels, "other")
    confidences = rng.uniform(size=(n, 3))
    rows, ranks = np.indices(labels.shape)
    path = tmp_path / "run.parquet"
    pq.write_table(
        pa.table(
            {
                "row": rows.ravel(),
                "level": ranks.ravel(),
                "label": labels.ravel(),
                "prediction": predictions.ravel(),
                "confidence": confidences.ravel(),
            }
        ),
        path,
    )
    weights = np.ones((n, 2), dtype=np.int8)
    weights[:8] = -1  # Calibration rows.
    weights[8, 1] = 3  # Drawn three times in replicate 1.
    np.save(tmp_path / "weights.npy", weights)
    aggregate, per_class, thresholds = replicates.evaluate_run(("run", path, tmp_path / "weights.npy", [0, 1]))
    values = pd.DataFrame(aggregate, columns=["run", "replicate", "setting", "level", "metric", "value"])

    def recall(replicate):
        return values.query("replicate == @replicate and setting == 'zero' and metric == 'recall' and level == 0").value.item()

    kwargs = dict(threshold=[0.0] * 3, simple=True, hierarchical=False, verbose=0)
    direct = evaluate_file(replicates.metric_frame(labels, predictions, confidences, np.arange(8, n)), **kwargs)
    assert recall(0) == direct["recall"][0]
    repeated = np.r_[np.arange(8, n), [8, 8]]
    assert recall(1) == evaluate_file(replicates.metric_frame(labels, predictions, confidences, repeated), **kwargs)["recall"][0]
    assert len(thresholds) == 2 * 3 and per_class


def test_parent_rules_differ_only_when_leaf_mass_is_split():
    # Two genera: g0 = {s0, s1}, g1 = {s2}. s2 wins, but g0 holds more summed mass.
    taxonomy = pd.DataFrame({"class_id": [0, 1, 2], "speciesKey": ["s0", "s1", "s2"], "genusKey": ["g0", "g0", "g1"], "familyKey": "f"})
    scores = pd.DataFrame(np.log([[0.3, 0.3, 0.4]]), columns=["c0", "c1", "c2"])
    summed, summed_conf = replicates.top1_predictions(scores, taxonomy, "leaf_sum")
    winner, winner_conf = replicates.top1_predictions(scores, taxonomy, "winner_ancestor")
    assert summed[0].tolist() == ["s2", "g0", "f"] and summed_conf[0] == pytest.approx([0.4, 0.6, 1.0])
    assert winner[0].tolist() == ["s2", "g1", "f"] and winner_conf[0] == pytest.approx([0.4, 0.4, 0.4])
