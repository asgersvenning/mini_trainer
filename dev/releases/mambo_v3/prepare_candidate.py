"""Prepare local release artifacts and an inventory. Never upload, tag or publish."""

import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
import tempfile
import tomllib
from pathlib import Path

from deployment.mambo_deploy.download import fetch_file
from dev.releases.mambo_v3.build_bundle import INPUTS, build
from dev.releases.mambo_v3.package_download_metadata import distribution_readme, package

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fetch_inputs(source):
    inventory = tomllib.loads((HERE / "inventory.toml").read_text())
    needed = {f"{inventory['production']}/{name}" for name in INPUTS.values()}
    for item in inventory["artifacts"]:
        if item["path"] in needed:
            fetch_file(item["url"], source / item["path"], size=item["size"], sha256=item["sha256"])


def prepare(source, output):
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT).strip():
        raise RuntimeError("Commit tracked release changes before preparing an identified candidate")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    output.mkdir(parents=True)
    bundle = output / "mambo-v3-bundle"
    build(source, bundle)
    dist = output / "dist"
    dist.mkdir()
    # Stage just the deployment project; no caches, experiments or training data.
    with tempfile.TemporaryDirectory(prefix="mambo-wheel-") as directory:
        stage = Path(directory)
        for name in ("pyproject.toml", "LICENSE"):
            shutil.copyfile(ROOT / "deployment" / name, stage / name)
        shutil.copytree(ROOT / "deployment/mambo_deploy", stage / "mambo_deploy", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        (stage / "README.md").write_text(distribution_readme())
        package(bundle, stage / "mambo_deploy/default_bundle.json")
        subprocess.run(["uv", "build", "--project", str(stage), "--wheel", "--sdist", "--out-dir", str(dist)], check=True)
    subprocess.run(["uv", "build", "--wheel", "--out-dir", str(dist)], cwd=ROOT, check=True)
    (output / "RELEASE_README.md").write_text(distribution_readme())
    for name in ("publication.md", "evidence-policy.md", "model-provenance.toml"):
        shutil.copyfile(HERE / name, output / name)
    evidence = output / "evidence"
    evidence.mkdir()
    # Public, compact evidence already committed; never sweep local-evidence inputs.
    tracked = subprocess.check_output(["git", "ls-files", "docs/assets/mambo-*"], cwd=ROOT, text=True).splitlines()
    for relative in tracked:
        source_file = ROOT / relative
        if source_file.suffix in {".json", ".csv", ".svg"}:
            shutil.copyfile(source_file, evidence / source_file.name)
    archive = output / "mambo-v3-bundle.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        stream.add(bundle, arcname=bundle.name)
    manifest = {
        "schema": "mambo-prepared-release-v1",
        "source_commit": commit,
        "distribution": "mambo-v3",
        "package_version": tomllib.loads((ROOT / "deployment/pyproject.toml").read_text())["project"]["version"],
        "model_id": "MAMBO_v3",
        "training_distribution": {key: tomllib.loads((ROOT / "pyproject.toml").read_text())["project"][key] for key in ("name", "version")},
        "publication_performed": False,
        "qualification": "Pending final installed-artifact checks; see qualification records alongside this manifest",
        "owner_decisions": [],
        "provenance_limits": [
            "Initialization lineage reconstructed from source; starting file hash and training Git revision not retained"
        ],
        "files": {},
    }
    for path in sorted(output.rglob("*")):
        if path.is_file():
            manifest["files"][path.relative_to(output).as_posix()] = {"size": path.stat().st_size, "sha256": digest(path)}
    (output / "release-candidate.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "SHA256SUMS").write_text(
        "".join(f"{entry['sha256']}  {name}\n" for name, entry in manifest["files"].items())
        + f"{digest(output / 'release-candidate.json')}  release-candidate.json\n"
    )
    print(output / "release-candidate.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Verified input inventory directory")
    parser.add_argument("--output", type=Path, required=True, help="New local artifact directory")
    parser.add_argument("--download", action="store_true", help="Fetch only pinned release inputs from ERDA")
    args = parser.parse_args()
    if args.download:
        fetch_inputs(args.source.resolve())
    prepare(args.source.resolve(), args.output.resolve())
