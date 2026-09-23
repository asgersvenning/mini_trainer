"""Exercise isolated prediction containers from the pinned legacy Git commit."""

import ast
import json
import subprocess
import tomllib
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent


def check():
    fixture = tomllib.loads((HERE / "compatibility.toml").read_text())
    namespace = dict(torch=torch, dataclass=dataclass, lru_cache=lru_cache, json=json, warnings=warnings)
    for path, names in (
        ("mini_trainer/classifier.py", {"PredictionItem", "BasePrediction"}),
        ("mini_trainer/hierarchical/model.py", {"HierarchicalPredictionItem", "HierarchicalPrediction"}),
    ):
        source = subprocess.check_output(["git", "show", f"{fixture['legacy_commit']}:{path}"], cwd=HERE, text=True)
        tree = ast.parse(source)
        tree.body = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in names]
        if {node.name for node in tree.body} != names:
            raise ValueError(f"Legacy container definitions missing: {path}")
        exec(compile(tree, path, "exec"), namespace)
    case = fixture["top1"]
    prediction = namespace["HierarchicalPrediction"](
        [torch.tensor([rank]) for rank in case["logits"]],
        cls2idx={str(rank): {label: index for index, label in enumerate(labels)} for rank, labels in enumerate(case["classes"])},
    )
    assert len(prediction) == 1
    assert list(prediction.indices.shape) == case["array_shape"]
    assert list(prediction.confidence.shape) == case["array_shape"]
    item = prediction[0]
    assert item.label == tuple(case["labels"])
    assert item.index == tuple(case["indices"])
    assert isinstance(item.confidence, tuple)
    torch.testing.assert_close(torch.tensor(item.confidence), torch.tensor(case["confidence"]))
    assert prediction.to_dict() == [{"label": item.label, "confidence": item.confidence, "index": item.index}]
    print("Pinned MAMBO_v2 top-1 container fixture passed (CPU; no model inference).")


if __name__ == "__main__":
    check()
