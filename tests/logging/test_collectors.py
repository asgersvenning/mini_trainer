import json

import pytest
import torch
from PIL import Image

from mini_trainer.hierarchical.integration import HierarchicalResultCollector
from mini_trainer.logging import BaseResultCollector


@pytest.mark.parametrize("hierarchical", [False, True], ids=["flat", "hierarchical"])
@pytest.mark.parametrize("count", [1, 5], ids=["single-observation", "multiple-observations"])
def test_evaluation_preserves_observations_and_per_rank_support(tmp_path, hierarchical, count):
    mappings = {"0": {"s1": 1, "s0": 0, "s2": 2}, "1": {"g1": 1, "g0": 0}}
    labels = [("s0", "g0"), ("s1", "g0"), ("s2", "g1"), ("unknown", "g1"), ("s0", "g0")][:count]
    species = torch.tensor([[8.0, 0.0, 0.0], [8.0, 0.0, 0.0], [0.0, 0.0, 8.0], [0.0, 8.0, 0.0], [8.0, 0.0, 0.0]])[:count]
    genus = torch.tensor([[8.0, 0.0], [8.0, 0.0], [0.0, 8.0], [0.0, 8.0], [0.0, 8.0]])[:count]
    cls = HierarchicalResultCollector if hierarchical else BaseResultCollector
    collector = cls(cls2idx=mappings if hierarchical else mappings["0"], scientific_names=False)
    collector.collect(
        paths=[f"{i}.jpg" for i in range(count)],
        predictions=[species, genus] if hierarchical else species,
        labels=labels if hierarchical else [row[0] for row in labels],
    )
    original = json.dumps(collector.data)
    results = collector.evaluate(str(tmp_path), prefix="sample_", plot_conf_mat=True)
    assert json.dumps(collector.data) == original
    assert json.loads((tmp_path / "sample_eval_results.json").read_text()) == json.loads(json.dumps(results))
    ranks = list(results.values()) if hierarchical else [results]
    species_result = ranks[0]
    assert list(species_result["conf_mat"]) == ["s0", "s1", "s2"]
    assert species_result["totals"] == ({"s0": 1, "s1": 0, "s2": 0} if count == 1 else {"s0": 2, "s1": 1, "s2": 1})
    assert species_result["conf_mat"]["s1"]["s0"] == (0 if count == 1 else 1)
    assert species_result["micro"] == pytest.approx(1 if count == 1 else 3 / 4)
    assert species_result["macro"] == pytest.approx(1 if count == 1 else 2 / 3)
    if hierarchical:
        assert ranks[1]["totals"] == ({"g0": 1, "g1": 0} if count == 1 else {"g0": 3, "g1": 2})
        assert ranks[1]["micro"] == pytest.approx(1 if count == 1 else 4 / 5)
        assert ranks[1]["macro"] == pytest.approx(1 if count == 1 else 5 / 6)
    for level in range(len(ranks)):
        suffix = f"_level{level}" if hierarchical else ""
        with Image.open(tmp_path / f"sample_confusion_matrix{suffix}.png") as image:
            image.verify()
