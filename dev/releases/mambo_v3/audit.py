"""Offline audit of pinned release inputs; does not construct models or use CUDA."""

import argparse
import csv
import hashlib
import json
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_files(root, inventory):
    """Fail closed on absent, changed, or escaping artifact paths."""
    root = root.resolve()
    for item in inventory["artifacts"]:
        path = (root / item["path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"Artifact escapes evidence root: {item['path']}")
        if path.stat().st_size != item["size"] or sha256(path) != item["sha256"]:
            raise ValueError(f"Artifact integrity mismatch: {item['path']}")


def recover_preset(state):
    mapping = state["classifier._extra_state"]["cls2idx"]["0"]
    reverse = {index: label for label, index in mapping.items()}
    indices = state["classifier.active_indices"].tolist()
    if len(indices) != len(set(indices)):
        raise ValueError("Duplicate preset indices")
    return [reverse[index] for index in indices]


def audit(root, flemming=None):
    import torch

    inventory = tomllib.loads((HERE / "inventory.toml").read_text())
    verify_files(root, inventory)
    production = root / inventory["production"]
    candidate = torch.load(production / "models/pytorch/best.pt", map_location="cpu", weights_only=True)
    metadata = candidate["classifier._extra_state"]
    manifest = json.loads((production / "models/onnx-fp32/manifest.json").read_text())
    if manifest["classifiers"][0]["metadata"]["cls2idx"] != metadata["cls2idx"]:
        raise ValueError("ONNX and PyTorch class ordering differs")
    full = torch.load(root / "MAMBO/hierarchical_bioclip2_ft_v1.pt", map_location="cpu", weights_only=True)
    states = [full]
    counts = {"full": len(metadata["cls2idx"]["0"])}
    for name, preset in inventory["presets"].items():
        state = torch.load(root / preset["source"], map_location="cpu", weights_only=True)
        states.append(state)
        path = HERE / preset["path"]
        labels = path.read_text().splitlines()
        if sha256(path) != preset["sha256"] or len(labels) != preset["count"] or labels != recover_preset(state):
            raise ValueError(f"Preset mismatch: {name}")
        counts[name] = len(labels)
    for state in states:
        if state["classifier._extra_state"]["cls2idx"] != metadata["cls2idx"]:
            raise ValueError("Legacy and candidate mappings differ")
        for rank in (0, 1):
            key = f"classifier.mask_{rank}"
            if not torch.equal(state[key], candidate[key]):
                raise ValueError(f"Legacy and candidate parent mapping differs: {key}")
    report = {"verified_files": len(inventory["artifacts"]), "presets": counts, "mapping_and_parent_order": "identical"}
    if flemming is not None:
        with (production / "evaluation/expert/predictions/mini_metric.csv").open() as stream:
            rows = [row for row in csv.DictReader(stream) if row["level"] == "0"]
        expected = {tuple(Path(row["filename"]).parts[-2:]) for row in rows}
        actual = {tuple(path.parts[-2:]) for path in flemming.glob("*/*.jpg")}
        if len(expected) != len(rows) or expected != actual:
            raise ValueError(f"Flemming identity mismatch: missing={len(expected - actual)}, extra={len(actual - expected)}")
        if any(row["label"] != Path(row["filename"]).parent.name for row in rows):
            raise ValueError("Flemming labels differ from species directories")
        report["flemming"] = {"images": len(expected), "species": len({key[0] for key in expected}), "identity": "species/path only"}
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--flemming", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.root, args.flemming), indent=2))
