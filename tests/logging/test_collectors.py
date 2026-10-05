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


def test_parquet_collector_streams_log_probabilities_embeddings_and_labels(tmp_path):
    pq = pytest.importorskip("pyarrow.parquet")
    from mini_trainer.logging import ParquetResultCollector
    from mini_trainer.modeling import EmbeddingContext

    collector = ParquetResultCollector(shard_size=3)
    torch.manual_seed(0)
    batches = []
    for start in (0, 4):
        species, genus, embeddings = torch.randn(4, 5), torch.randn(4, 2), torch.nn.functional.normalize(torch.randn(4, 3), dim=1)
        with EmbeddingContext():
            EmbeddingContext.set(embeddings)
            collector.collect(
                paths=[f"{start + i}.jpg" for i in range(4)],
                predictions=[species, genus],
                labels=[[start + i, i % 2] for i in range(4)],
            )
        batches.append((species, genus, embeddings))
    collector.save(str(tmp_path / "out"))

    out = tmp_path / "out"
    assert sorted(p.name for p in (out / "rank-0").iterdir()) == ["part-00000.parquet", "part-00001.parquet"]
    index = pq.read_table(out / "index.parquet").to_pandas()
    assert index.row.tolist() == list(range(8)) and index.path.tolist() == [f"{i}.jpg" for i in range(8)]
    assert index.label_0.tolist() == list(range(8)) and index.label_1.tolist() == [0, 1] * 4
    for name, position in (("rank-0", 0), ("rank-1", 1), ("embeddings", 2)):
        table = pq.read_table(out / name).to_pandas().sort_values("row")
        expected = torch.cat([batch[position] for batch in batches])
        if name != "embeddings":
            expected = torch.log_softmax(expected, -1)
        values = torch.tensor(table.drop(columns="row").to_numpy(), dtype=torch.float32)
        torch.testing.assert_close(values, expected, atol=2e-3, rtol=1e-3)  # float16 storage
    with pytest.raises(FileExistsError):
        ParquetResultCollector().save(str(out))
