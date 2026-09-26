"""Explicit publication entry point; workflows call it only after human release gates."""

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

from dev.releases.mambo_v3.prepare_candidate import digest


def verified_payload(folder):
    payload = json.loads((folder / "publication.json").read_text())
    paths = {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()}
    if paths != set(payload["files"]) | {"publication.json"}:
        raise ValueError("Staged publication file set differs from its inventory")
    for relative, expected in payload["files"].items():
        path = folder / relative
        if not path.resolve().is_relative_to(folder.resolve()) or path.is_symlink() or digest(path) != expected:
            raise ValueError(f"Staged publication integrity mismatch: {relative}")
    return payload


def github(folder, repository, tag):
    verified_payload(folder)
    release = json.loads(subprocess.check_output(["gh", "api", f"repos/{repository}/releases/tags/{tag}"], text=True))
    existing = {item["name"]: item for item in release["assets"]}
    for path in sorted(folder.iterdir()):
        if not path.is_file():
            raise ValueError("GitHub release assets must be flat files")
        if asset := existing.get(path.name):
            with tempfile.TemporaryFile() as stream:
                subprocess.run(
                    ["gh", "api", "-H", "Accept: application/octet-stream", f"repos/{repository}/releases/assets/{asset['id']}"],
                    stdout=stream,
                    check=True,
                )
                stream.seek(0)
                import hashlib

                if hashlib.file_digest(stream, "sha256").hexdigest() != digest(path):
                    raise ValueError(f"Refusing to replace published asset: {path.name}")
        else:
            subprocess.run(["gh", "release", "upload", tag, str(path), "--repo", repository], check=True)


def hub(folder, repository, kind):
    from huggingface_hub import HfApi, hf_hub_download

    payload = verified_payload(folder)
    api = HfApi(token=os.environ["HF_TOKEN"])
    tag = f"v{payload['package_version']}" if kind == "model" else f"source-{payload['source_commit']}"
    # Account/repository creation is a separate owner task, not implicit here.
    tags = {ref.name for ref in api.list_repo_refs(repository, repo_type=kind).tags}
    if tag in tags:
        existing = Path(hf_hub_download(repository, "publication.json", repo_type=kind, revision=tag, token=api.token))
        if json.loads(existing.read_text()) != payload:
            raise ValueError(f"Refusing to replace published {kind} revision: {tag}")
        print(f"Already published: {repository}@{tag}")
        return
    commit = api.upload_folder(repo_id=repository, repo_type=kind, folder_path=folder, commit_message=f"Publish {tag}")
    api.create_tag(repo_id=repository, repo_type=kind, tag=tag, revision=commit.oid)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("github", "model", "space"))
    parser.add_argument("folder", type=Path)
    parser.add_argument("repository")
    parser.add_argument("--tag", default="MAMBO_v3")
    args = parser.parse_args()
    if args.kind == "github":
        github(args.folder.resolve(), args.repository, args.tag)
    else:
        hub(args.folder.resolve(), args.repository, args.kind)
