"""Candidate filtering must not filter evaluation samples or their labels."""

import json

import pytest
import torch

from mini_trainer.hierarchical.model import HierarchicalClassifier
from mini_trainer.modeling import Classifier
from mini_trainer.modeling.mask import restrict_class_labels
from mini_trainer.predict import cli, main


def head(hierarchical):
    kwargs = {"cls2idx": {"a": 0, "b": 1, "c": 2}}
    if hierarchical:
        kwargs = {"cls2idx": {"0": kwargs["cls2idx"], "1": {"g": 0, "h": 1}}, "sparse_masks": [torch.tensor([0, 0, 1])]}
    cls = HierarchicalClassifier if hierarchical else Classifier
    return cls(in_features=4, out_features=3, hidden=False, normalized=True, **kwargs).eval()


@pytest.mark.parametrize("hierarchical", [False, True])
def test_filter_preserves_weight_indices_and_parent_mapping(hierarchical):
    model = head(hierarchical)
    model.set_active_features([0, 2])
    wrapped = torch.nn.Sequential(model)
    report = restrict_class_labels(wrapped, ["c", "c", "missing"])
    assert model.active_indices.tolist() == [2]
    assert report["retained_labels"] == ["c"]
    assert report["missing_labels"] == ["missing"]
    assert report["excluded_labels"] == ["a"]
    expected = {"0": {"c": 0}, "1": {"h": 0}} if hierarchical else {"c": 0}
    assert model.metadata["cls2idx"] == expected
    result = model(torch.randn(2, 4))
    assert all(t.shape == (2, 1) for t in (result if hierarchical else [result]))
    with pytest.raises(ValueError, match="no overlap"):
        restrict_class_labels(wrapped, ["absent"])
    assert model.active_indices.tolist() == [2]


def test_cli_class_list_overrides_yaml(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    config.write_text("input: images\nweights: model.pt\nclass_list: old.txt\n")
    monkeypatch.setattr("sys.argv", ["mt_predict", "--config", str(config), "--class-list", "new.txt"])
    assert cli()["class_list"] == "new.txt"


def test_inference_keeps_excluded_ground_truth(tmp_path, monkeypatch):
    import mini_trainer.predict as module

    model = head(True)
    received = []

    class Builder:
        @staticmethod
        def build_model(**kwargs):
            return torch.nn.Sequential(model), lambda x: x

        @staticmethod
        def build_inference_dataloader(images, **kwargs):
            assert images == ["one.jpg", "two.jpg"]
            return torch.utils.data.DataLoader(torch.randn(2, 4), batch_size=2)

    class Collector:
        def __init__(self, model, **kwargs):
            assert model[0].metadata["cls2idx"]["0"] == {"c": 0}

        def collect(self, **kwargs):
            received.extend(kwargs["labels"])

        def save(self, *args, **kwargs):
            pass

    def metadata(*args, **kwargs):
        assert set(kwargs["cls2idx"]["0"]) == {"a", "b", "c"}
        return {"path": ["one.jpg", "two.jpg"], "split": ["test", "test"], "label": [["a", "g"], ["unknown", "h"]]}

    monkeypatch.setattr(module, "get_metadata", metadata)
    monkeypatch.setattr(module, "dump_resolved_config", lambda **kwargs: None)
    class_list = tmp_path / "classes.txt"
    class_list.write_text("c\n\nc\nmissing\n")
    main(
        input=str(tmp_path),
        weights="unused",
        output=str(tmp_path),
        name="run",
        device="cpu",
        dtype="float32",
        builder=Builder,
        collector_cls=Collector,
        data_index="unused",
        class_list=str(class_list),
    )
    assert received == [["a", "g"], ["unknown", "h"]]
    report = json.loads((tmp_path / "run/class_filter.json").read_text())
    assert report["retained_count"] == 1
    assert len(report["sha256"]) == 64
