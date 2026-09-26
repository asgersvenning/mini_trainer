"""Assemble verified release inputs into a relocatable, offline candidate bundle."""

import argparse
import hashlib
import json
import shutil
import tempfile
import tomllib
from pathlib import Path

from deployment.mambo_deploy.preprocessing import RECIPE
from dev.releases.mambo_v3.audit import HERE, sha256
from dev.releases.mambo_v3.package_download_metadata import distribution_readme

INPUTS = {
    "models/pytorch/best.pt": "models/pytorch/best.pt",
    "models/onnx/model.onnx": "models/onnx-fp32/model.onnx",
    "models/onnx/model.onnx.data": "models/onnx-fp32/model.onnx.data",
    "models/onnx/manifest.json": "models/onnx-fp32/manifest.json",
    "models/onnx-embedding/model.onnx": "viewer/browser-model/model.onnx",
    "models/onnx-embedding/model.onnx.data": "viewer/browser-model/model.onnx.data",
    "models/onnx-embedding/manifest.json": "viewer/browser-model/manifest.json",
}


def build(source, destination):
    if destination.exists():
        raise FileExistsError(f"Refusing to replace existing bundle: {destination}")
    inventory = tomllib.loads((HERE / "inventory.toml").read_text())
    presets = tomllib.loads((HERE / "preset-manifest.toml").read_text())
    definitions = tomllib.loads((HERE / "preset-definitions.toml").read_text())
    provenance = tomllib.loads((HERE / "model-provenance.toml").read_text())
    if sha256(HERE / "preset-definitions.toml") != presets["definitions_sha256"]:
        raise ValueError("Preset manifest is stale; rebuild presets first")
    files = {item["path"]: item for item in inventory["artifacts"]}
    production = inventory["production"]

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mambo-bundle-", dir=destination.parent) as temp:
        root = Path(temp) / "bundle"
        root.mkdir()
        origins = {}
        for target, relative in INPUTS.items():
            item = files[f"{production}/{relative}"]
            path = source / item["path"]
            if path.stat().st_size != item["size"] or sha256(path) != item["sha256"]:
                raise ValueError(f"Input integrity mismatch: {path}")
            output = root / target
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, output)
            origins[target] = item["url"]
        export = json.loads((root / "models/onnx/manifest.json").read_text())
        metadata = export["classifiers"][0]["metadata"]
        if metadata.get("prior") is not None or metadata.get("normalized") is not True:
            raise ValueError("Unqualified head semantics")
        mappings = metadata["cls2idx"]
        labels = [sorted(mappings[str(rank)], key=mappings[str(rank)].get) for rank in range(3)]
        parents = [[-1] * len(labels[rank]) for rank in range(2)]
        for species, lineage in metadata["labels"].items():
            for rank in range(2):
                child = mappings[str(rank)][lineage[rank]]
                parent = mappings[str(rank + 1)][lineage[rank + 1]]
                if parents[rank][child] not in (-1, parent):
                    raise ValueError(f"Ambiguous parent for {species}")
                parents[rank][child] = parent
        if any(-1 in rank for rank in parents):
            raise ValueError("Incomplete class hierarchy")

        def write_json(relative, data):
            (root / relative).write_text(json.dumps(data, indent=2) + "\n")

        write_json("classes.json", {"ranks": ["species", "genus", "family"], "labels": labels, "parents": parents})
        write_json("preprocessing.json", RECIPE)
        regions = {}
        (root / "regions").mkdir()
        for name, item in presets["presets"].items():
            path = HERE / item["path"]
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != item["sha256"]:
                raise ValueError(f"Preset integrity mismatch: {name}")
            target = f"regions/{name}.classes"
            (root / target).write_bytes(data)
            regions[name] = {
                **item,
                "path": target,
                "scope": definitions["presets"][name]["scope"],
                "qualification_status": definitions["qualification_status"],
            }
        write_json("presets.json", regions)
        shutil.copyfile(HERE / "preset-definitions.toml", root / "PRESET_DEFINITIONS.toml")
        shutil.copyfile(HERE / "preset-updates.toml", root / "PRESET_UPDATES.toml")
        (root / "README.md").write_text(distribution_readme())
        shutil.copyfile(HERE.parents[2] / "LICENSE", root / "CODE_LICENSE")
        lines = [
            "# Presets",
            "",
            "V3 uses metadata row counts, including all splits, without further deduplication. "
            "New lists require at least 3 regional and 25 global rows; legacy lists preserve their historical membership.",
            "",
            "| Preset | Species | Regional/global minimum rows | Scope |",
            "|---|---:|---|---|",
        ]
        lines += [
            f"| {name} | {item['count']} | {item['minimum_regional_rows']}/{item['minimum_global_rows'] or 'none'} | {item['scope']} |"
            for name, item in regions.items()
        ]
        lines += ["", "Exact filters: PRESET_DEFINITIONS.toml. `full` includes all 12,632 model species.", ""]
        (root / "PRESETS.md").write_text("\n".join(lines))
        for filename in ("MODEL_CARD.md", "NOTICES.md", "MODEL_LICENSE.txt"):
            shutil.copyfile(HERE / filename, root / filename)
        shutil.copyfile(HERE / "model-provenance.toml", root / "MODEL_PROVENANCE.toml")
        profiles = {
            "torch": {"model": "models/pytorch/best.pt", "files": ["models/pytorch/best.pt"]},
            "onnx": {"model": "models/onnx/model.onnx", "files": ["models/onnx/model.onnx", "models/onnx/model.onnx.data"]},
            "onnx-embedding": {
                "model": "models/onnx-embedding/model.onnx",
                "files": ["models/onnx-embedding/model.onnx", "models/onnx-embedding/model.onnx.data"],
            },
        }
        manifest = {
            "schema": "mambo-release-v1",
            "model_id": "MAMBO_v3",
            "artifact_revision": 3,
            "package_version": "0.3.0",
            "distribution": "mambo-v3",
            "default_preset": "full",
            "licenses": {
                "code": "MIT",
                "weights": provenance["weights_license"],
                "weights_text": "MODEL_LICENSE.txt",
                "notices": "NOTICES.md",
            },
            "score_semantics": "hierarchical-leaf-logits-logsumexp-v1",
            "profiles": profiles,
            "embedding": {"dimension": 1280, "stage": "normalized preclassification"},
            "origins": origins,
            "files": {},
        }
        for path in sorted(root.rglob("*")):
            if path.is_file():
                manifest["files"][path.relative_to(root).as_posix()] = {"size": path.stat().st_size, "sha256": sha256(path)}
        write_json("release.json", manifest)
        root.rename(destination)
    print(destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    build(args.source, args.destination)
