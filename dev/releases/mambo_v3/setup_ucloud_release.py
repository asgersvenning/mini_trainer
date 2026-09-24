"""Prepare cached public models and the original test split from one Parquet path."""

import argparse
import io
import json
import os
import socket
import subprocess
import sys
import tarfile
import tomllib
from pathlib import Path

from deployment.mambo_deploy.bundle import Bundle
from deployment.mambo_deploy.download import default_bundle, fetch_file
from dev.benchmarks.inference.onnx_inference import file_hash
from dev.releases.mambo_v3.audit import HERE
from dev.releases.mambo_v3.evaluation_data import write_json
from dev.releases.mambo_v3.legacy_evaluation import COMMIT
from dev.releases.mambo_v3.prepare_ucloud import recover


def setup(args):
    metadata = args.metadata.resolve()
    expected = tomllib.loads((HERE / "construction.toml").read_text())["source"]["sha256"]
    if file_hash(metadata) != expected:
        raise ValueError("Original metadata snapshot hash mismatch")
    root = (args.root or metadata.parent).resolve()
    cache = args.cache.expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["MAMBO_CACHE"] = str(cache / "models")
    if args.offline:
        os.environ["MAMBO_OFFLINE"] = "1"
    bundle_path = default_bundle()
    bundle = Bundle(bundle_path, download=True)
    for relative in bundle.manifest["files"]:
        bundle.file(relative)
    inventory = tomllib.loads((HERE / "inventory.toml").read_text())
    production = inventory["production"]
    suffixes = ("evaluation/in-domain/provenance/staging.json", "evaluation/in-domain/predictions/mini_metric.csv")
    needed = [a for a in inventory["artifacts"] if a["path"].startswith("MAMBO/") or a["path"] in [f"{production}/{p}" for p in suffixes]]
    for item in needed:
        fetch_file(item["url"], cache / "archives" / item["path"], size=item["size"], sha256=item["sha256"], offline=args.offline)
    source = cache / "legacy-source" / COMMIT
    if not source.exists():
        source.parent.mkdir(parents=True, exist_ok=True)
        archive = subprocess.check_output(["git", "archive", COMMIT, "mini_trainer"], cwd=HERE)
        source.mkdir()
        with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
            stream.extractall(source, filter="data")
    from huggingface_hub import hf_hub_download

    hf_cache = cache / "huggingface/hub"
    for name, digest in {
        "open_clip_config.json": "1bf947e96e943fe50efd5c3e26c37f843a2fa3c358967719a68c8a6d17ce68c8",
        "open_clip_model.safetensors": "b7b2bf6fbc95799e42630e394cf95803892ab447c1a8ab629dbc82fbeaf7dfef",
    }.items():
        path = hf_hub_download(
            "imageomics/bioclip-2",
            name,
            revision="2957b322090f9cb17ae72c71981c7218a28d81e0",
            cache_dir=hf_cache,
            local_files_only=args.offline,
        )
        if file_hash(path) != digest:
            raise ValueError("Unexpected BioCLIP-2 backbone")
    # The legacy API requests the default branch; point its offline ref to the verified snapshot.
    refs = hf_cache / "models--imageomics--bioclip-2/refs"
    refs.mkdir(parents=True, exist_ok=True)
    (refs / "main").write_text("2957b322090f9cb17ae72c71981c7218a28d81e0")
    staging, reference = [cache / "archives" / production / suffix for suffix in suffixes]
    manifest = cache / "global-lepi-test-manifest.json"
    provenance = {k: file_hash(v) for k, v in [("metadata", metadata), ("staging", staging), ("reference", reference)]}
    provenance["test_set"] = "0"
    if manifest.exists():
        prior = json.loads(manifest.read_text())
        if prior.get("provenance") != provenance or len(prior["records"]) != 632913:
            raise ValueError("Cached manifest differs from original split provenance")
    else:
        records = recover(metadata, staging, reference)
        if len(records) != 632913:
            raise ValueError("Expected original 632,913-image test split")
        for i, record in enumerate(records):
            path = (root / record["path"]).resolve()
            if not path.is_relative_to(root):
                raise ValueError("Unsafe image path")
            record["sha256"] = file_hash(path)
            if i % 10000 == 0:
                print(f"Hashed {i}/{len(records)} test images", flush=True)
        temporary = manifest.with_suffix(".partial.json")
        write_json(temporary, {"schema_version": 1, "dataset": "global-lepi-test", "provenance": provenance, "records": records})
        temporary.replace(manifest)
    config = json.loads((HERE / "ucloud_release.json").read_text())
    config.update(
        environment_id=socket.gethostname(),
        v2_python=str(args.v2_python or sys.executable),
        v3_python=sys.executable,
        metrics_python=str(args.metrics_python or sys.executable),
        legacy_source=str(source),
        legacy_weights=str(cache / "archives/MAMBO"),
        hf_cache=str(hf_cache),
        bundle=str(bundle_path),
        manifest=str(manifest),
        root=str(root),
        output=str(cache / "runs"),
    )
    output = cache / "ucloud-release.json"
    if output.exists() and json.loads(output.read_text()) != config:
        raise ValueError("Existing run configuration differs; preserve it or choose a new cache")
    write_json(output, config)
    print(f"Ready: {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--root", type=Path, help="Defaults to the Parquet parent; expects images/species/filename below it")
    parser.add_argument("--cache", type=Path, default=Path.home() / ".cache/mambo-ucloud")
    parser.add_argument("--v2-python", type=Path)
    parser.add_argument("--metrics-python", type=Path)
    parser.add_argument("--offline", action="store_true")
    setup(parser.parse_args())
