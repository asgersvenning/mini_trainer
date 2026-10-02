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
    provenance = tomllib.loads(Path(__file__).with_name("model-provenance.toml").read_text())
    facts = {
        "vocabulary": f"{counts[0]}, {counts[1]} and {counts[2]}",
        "embedding_dim": f"{manifest['embedding']['dimension']:,}",
        "epochs": str(provenance["training_epochs"]),
        "training_images": f"{provenance['training_images']:,}",
        "training_config": provenance["configuration"]["url"],
        "performance": card_performance(),
    }
    for key, value in facts.items():
        template = template.replace("{{" + key + "}}", value)
    return template


def card_performance(directory=None, *, required=False):
    """A smoke run or edited figure must never become published comparison evidence."""
    directory = Path(directory) if directory else Path(__file__).with_name("card-performance")
    if not (directory / "summary.json").exists():
        if required:
            raise ValueError("Run and review the 5,000-image iNaturalist comparison before publication")
        return "The maintenance comparison is awaiting its UCloud run; no new results are claimed."
    summary = json.loads((directory / "summary.json").read_text())
    if summary["count"] != 5000 or summary["unmapped_truth"]:
        raise ValueError("Card requires 5,000 images and resolved ground-truth taxonomy")
    if set(summary["models"]) != {"nemo", "meghan", "inaturalist"}:
        raise ValueError("Card comparison is incomplete")
    for name in ("quality.png", "speed.png"):
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != summary["figures"][name]:
            raise ValueError(f"Changed card figure: {name}")
    return (
        "![Macro-Accuracy and Macro-F1](performance/quality.png)\n\n"
        "![Prediction speed](performance/speed.png)\n\n"
        f"5,000 recent Research Grade Lepidoptera observations, frozen {summary['cutoff'][:10]}; "
        "one image per observation. Species-level macro metrics from `mini_metrics`, "
        "global vocabulary, no TTA, no location input. Meghan is MAMBO_v2. "
        "Speed: Nemo ONNX and Meghan PyTorch on the same CPU; iNaturalist includes network latency. "
        "[Method and results](performance/summary.json)."
    )


def package(source, output):
    bundle = Bundle(source)
    metadata = {}
    # Model-card figures belong to Hub/offline assets, not the runtime's text bootstrap.
    for relative in list(bundle.manifest["files"]):
        if relative.startswith("performance/"):
            del bundle.manifest["files"][relative]
            continue
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
