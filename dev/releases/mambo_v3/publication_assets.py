"""Stage/verify explicit public release files; never upload or publish anything."""

import argparse
import json
import shutil
import subprocess
import tarfile
import tomllib
from pathlib import Path

from dev.releases.mambo_v3.prepare_candidate import HERE, ROOT, digest


def verify(candidate, *, qualified=True):
    manifest = json.loads((candidate / "release-candidate.json").read_text())
    project = tomllib.loads((ROOT / "deployment/pyproject.toml").read_text())["project"]
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if (manifest["distribution"], manifest["package_version"], manifest["model_id"], manifest["source_commit"]) != (
        project["name"],
        project["version"],
        "MAMBO_v3",
        source,
    ):
        raise ValueError("Candidate identity differs from the checked-out release")
    bundle = json.loads((candidate / "mambo-v3-bundle/release.json").read_text())
    if any(bundle[key] != manifest[key] for key in ("model_id", "package_version", "distribution")):
        raise ValueError("Bundle and package release identities differ")
    for relative, entry in manifest["files"].items():
        path = candidate / relative
        if not path.resolve().is_relative_to(candidate.resolve()) or path.is_symlink():
            raise ValueError(f"Invalid candidate path: {relative}")
        if path.stat().st_size != entry["size"] or digest(path) != entry["sha256"]:
            raise ValueError(f"Candidate integrity mismatch: {relative}")
    actual = {p.relative_to(candidate).as_posix() for p in candidate.rglob("*") if p.is_file()}
    if actual != set(manifest["files"]) | {"release-candidate.json", "SHA256SUMS"}:
        raise ValueError("Candidate contains unaccounted or missing files")
    if qualified and manifest["qualification"] != "passed":
        raise ValueError("Installed-artifact qualification is not complete")
    return manifest


def seal(candidate):
    manifest = json.loads((candidate / "release-candidate.json").read_text())
    manifest["files"] = {
        path.relative_to(candidate).as_posix(): {"size": path.stat().st_size, "sha256": digest(path)}
        for path in sorted(candidate.rglob("*"))
        if path.is_file() and path.name not in {"release-candidate.json", "SHA256SUMS"}
    }
    (candidate / "release-candidate.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (candidate / "SHA256SUMS").write_text(
        "".join(f"{entry['sha256']}  {name}\n" for name, entry in manifest["files"].items())
        + f"{digest(candidate / 'release-candidate.json')}  release-candidate.json\n"
    )


def stage_space(output, source=None, version=None):
    source = source or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    version = version or tomllib.loads((ROOT / "deployment/pyproject.toml").read_text())["project"]["version"]
    output.mkdir(parents=True, exist_ok=False)
    for name in ("app.py", "README.md", "requirements.txt", "taxon-names.json"):
        shutil.copyfile(ROOT / "deployment/demo" / name, output / name)
    (output / "publication.json").write_text(
        json.dumps(
            {
                "source_commit": source,
                "model_id": "MAMBO_v3",
                "package_version": version,
                "files": {p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file()},
            },
            indent=2,
        )
        + "\n"
    )
    return output


def stage(candidate, output):
    manifest = verify(candidate)
    output.mkdir(parents=True, exist_ok=False)
    model, space, github = (output / name for name in ("model", "space", "github"))
    model.mkdir()
    github.mkdir()
    shutil.copytree(candidate / "mambo-v3-bundle", model / "bundle")
    stage_space(space, manifest["source_commit"], manifest["package_version"])
    shutil.copyfile(HERE / "MODEL_CARD.md", model / "README.md")
    for name in ("MODEL_LICENSE.txt", "NOTICES.md"):
        shutil.copyfile(HERE / name, model / name)
    shutil.copyfile(ROOT / "deployment/CITATION.cff", model / "CITATION.cff")
    shutil.copyfile(candidate / "release-candidate.json", model / "release-candidate.json")
    # GitHub gets the offline bundle, public evidence and deployment distributions.
    # The training wheel is installed for qualification but has its own publication.
    for name in ("mambo-v3-bundle.tar.gz", "RELEASE_README.md", "release-candidate.json"):
        shutil.copyfile(candidate / name, github / name)
    for path in (candidate / "dist").glob("mambo_v3-*"):
        shutil.copyfile(path, github / path.name)
    with tarfile.open(github / "mambo-v3-evidence.tar.gz", "w:gz") as archive:
        for name in ("evidence", "qualification", "publication.md", "evidence-policy.md", "model-provenance.toml"):
            archive.add(candidate / name, arcname=name)
    (github / "SHA256SUMS").write_text("".join(f"{digest(p)}  {p.name}\n" for p in sorted(github.iterdir()) if p.is_file()))
    for folder in (model, space, github):
        (folder / "publication.json").unlink(missing_ok=True)
        payload = {
            "source_commit": manifest["source_commit"],
            "model_id": manifest["model_id"],
            "package_version": manifest["package_version"],
            "files": {p.relative_to(folder).as_posix(): digest(p) for p in sorted(folder.rglob("*")) if p.is_file()},
        }
        (folder / "publication.json").write_text(json.dumps(payload, indent=2) + "\n")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output:
        stage(args.candidate.resolve(), args.output.resolve())
    else:
        verify(args.candidate.resolve())
