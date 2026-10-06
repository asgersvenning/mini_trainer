"""Hierarchy experiments preserve complete groups and a matched species control."""

import numpy as np
import pandas as pd
import pytest
import torch

from mini_trainer.hierarchical.model import HierarchicalClassifier
from mini_trainer.modeling import Classifier
from publication.experiments.training_ablations.data import hierarchy_spec, select_families, table_digest, write_json
from publication.experiments.training_ablations.training import StudyBuilder


def test_family_selection_and_rank_counts_ignore_row_order():
    table = pd.DataFrame(
        {"familyKey": ["a", "a", "b", "b", "b", "c"], "genusKey": ["x", "x", "y", "y", "z", "w"], "train": [2, 9, 3, 6, 4, 1]},
        index=list("ABCDEF"),
    )
    families = select_families(table, 20, 42)
    assert families == select_families(table.sample(frac=1, random_state=12), 20, 42)
    selected = table[table.familyKey.isin(families)]
    assert selected.train.sum() <= 20
    spec = hierarchy_spec(selected, sorted(selected.index))
    assert [sum(counts) for counts in spec["counts"]] == [int(selected.train.sum())] * 3
    for source, target, mask in zip(spec["counts"], spec["counts"][1:], spec["masks"]):
        np.testing.assert_array_equal(np.bincount(mask, weights=source), target)


@pytest.mark.parametrize("loss_name", ["ce", "fixed", "emla"])
def test_species_control_matches_flat_gradient(tmp_path, monkeypatch, loss_name):
    write_json(
        tmp_path / "classes.json",
        {"num_classes": 4, "counts": [1, 4, 5, 8], "hierarchy": {"num_classes": [4, 3, 2], "counts": [[1, 4, 5, 8], [5, 5, 8], [10, 8]]}},
    )
    monkeypatch.setattr(StudyBuilder, "root", tmp_path)
    monkeypatch.setattr(StudyBuilder, "config", {"hierarchy": True})
    monkeypatch.setattr(StudyBuilder, "run", {"loss": loss_name})
    flat_criterion = StudyBuilder.build_criterion(device="cpu")
    monkeypatch.setattr(StudyBuilder, "run", {"loss": loss_name, "rank_weights": [1, 0, 0]})
    criterion = StudyBuilder.build_criterion(device="cpu")
    torch.manual_seed(42)
    flat = Classifier(8, 4, hidden=False, skip_spherical_init=True)
    torch.manual_seed(42)
    hierarchical = HierarchicalClassifier(
        in_features=8,
        out_features=4,
        hidden=False,
        skip_spherical_init=True,
        sparse_masks=[torch.tensor([0, 0, 1, 2]), torch.tensor([0, 0, 1])],
    )
    x = torch.randn(4, 8)
    target = torch.tensor([[0, 0, 0], [1, 0, 0], [2, 1, 0], [3, 2, 1]])
    leaf = flat(x)
    ranks = hierarchical(x)
    torch.testing.assert_close(leaf, ranks[0], rtol=0, atol=0)
    loss = flat_criterion(leaf, target[:, 0])
    control = sum(criterion(ranks, target))
    loss.backward()
    control.backward()
    torch.testing.assert_close(loss, control, rtol=0, atol=0)
    torch.testing.assert_close(
        flat.linear.parametrizations.weight.original1.grad, hierarchical.linear.parametrizations.weight.original1.grad, rtol=0, atol=0
    )


def test_cohort_content_digest_preserves_values_order_and_columns():
    frame = pd.DataFrame({"label": [0, 1], "path": ["a,b", "c\nd"]})
    assert table_digest(frame) == table_digest(frame.copy())
    assert table_digest(frame) != table_digest(frame.iloc[::-1])
    assert table_digest(frame) != table_digest(frame.rename(columns={"path": "other"}))
    changed = frame.copy()
    changed.loc[0, "path"] = "ab"
    assert table_digest(frame) != table_digest(changed)
