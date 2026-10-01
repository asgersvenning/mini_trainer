"""Freeze a verified local bundle's small metadata for automatic ERDA bootstrap."""

import argparse
import hashlib
import json
import re
import tomllib
from pathlib import Path

from deployment.mambo_deploy.bundle import Bundle


def distribution_readme(ref="models/mambo-v3/v0.3.1"):
    """Keep documentation links meaningful in PyPI metadata and extracted bundles."""
    root = Path(__file__).resolve().parents[3]
    readme = root / "deployment/README.md"

    def link(match):
        prefix, target = match.groups()
        if "://" in target or target.startswith("#"):
            return match.group(0)
        filename, separator, anchor = target.partition("#")
        relative = (readme.parent / filename).resolve().relative_to(root).as_posix()
        base = (
            "https://raw.githubusercontent.com/asgersvenning/mini_trainer/"
            if prefix.startswith("!")
            else "https://github.com/asgersvenning/mini_trainer/blob/"
        )
        return f"{prefix}({base}{ref}/{relative}{separator}{anchor})"

    return re.sub(r"(!?\[[^\]]*\])\(([^)]+)\)", link, readme.read_text())


def native_descriptor(metadata):
    """Derive native bootstrap data from the same manifest used by ONNX."""
    manifest = json.loads(metadata["release.json"])
    checkpoint = manifest["profiles"]["torch"]["model"]
    presets = json.loads(metadata["presets.json"])
    return {
        "url": manifest["origins"][checkpoint],
        "state_sha256": tomllib.loads(Path(__file__).with_name("model-provenance.toml").read_text())["checkpoint"]["state_sha256"],
        **manifest["files"][checkpoint],
        "preprocessing": json.loads(metadata["preprocessing.json"]),
        "presets": {name: metadata[item["path"]].splitlines() for name, item in presets.items()},
    }


def model_card(manifest, classes):
    """Render public facts from the bundle; do not maintain parallel counts."""
    template = Path(__file__).with_name("MODEL_CARD.md").read_text()
    counts = [f"{len(labels):,} {rank}" for labels, rank in zip(classes["labels"], ("species", "genera", "families"), strict=True)]
    return template.replace("{{vocabulary}}", f"{counts[0]}, {counts[1]} and {counts[2]}").replace(
        "{{embedding_dim}}", f"{manifest['embedding']['dimension']:,}"
    )


def package(source, output):
    bundle = Bundle(source)
    metadata = {}
    for relative in bundle.manifest["files"]:
        if relative not in bundle.manifest["origins"]:
            metadata[relative] = bundle.file(relative).read_text()
    # Ship current integration guidance, not the README frozen in the local bundle.
    metadata["README.md"] = distribution_readme()
    data = metadata["README.md"].encode()
    bundle.manifest["files"]["README.md"] = {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    metadata["release.json"] = json.dumps(bundle.manifest, indent=2) + "\n"
    output.write_text(json.dumps({"metadata": metadata}, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    package(args.source, args.output)
