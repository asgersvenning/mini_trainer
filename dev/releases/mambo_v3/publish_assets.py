"""Explicit publication entry point; workflows call it only after human release gates."""

import argparse
import json
import os
import re
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
    github_files(sorted(folder.iterdir()), repository, tag)


def github_files(paths, repository, tag):
    release = json.loads(subprocess.check_output(["gh", "api", f"repos/{repository}/releases/tags/{tag}"], text=True))
    existing = {item["name"]: item for item in release["assets"]}
    for path in paths:
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
    from huggingface_hub.errors import EntryNotFoundError

    payload = verified_payload(folder)
    api = HfApi(token=os.environ["HF_TOKEN"])
    # Resolve main once, then read at that immutable commit. Missing repositories
    # and authorization failures must propagate; only a missing manifest is new.
    revision = api.repo_info(repository, repo_type=kind).sha
    existing = None
    if revision:
        try:
            manifest = hf_hub_download(repository, "publication.json", repo_type=kind, revision=revision, token=api.token)
        except EntryNotFoundError:
            pass
        else:
            existing = json.loads(Path(manifest).read_text())
    if existing != payload:
        if existing is not None and (kind == "model" or existing["source_commit"] == payload["source_commit"]):
            raise ValueError(f"Refusing to replace published {kind} payload")
        commit = api.upload_folder(
            repo_id=repository,
            repo_type=kind,
            folder_path=folder,
            commit_message=f"Publish {payload['model_id']} from {payload['source_commit']}",
            parent_commit=revision,
        )
        revision = commit.oid
    if not revision or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Hub did not return an immutable commit SHA")
    receipt = {
        "repository": repository,
        "repo_type": kind,
        "revision": revision,
        "source_commit": payload["source_commit"],
        "model_id": payload["model_id"],
        "package_version": payload["package_version"],
        "publication_sha256": digest(folder / "publication.json"),
    }
    prefix = "spaces/" if kind == "space" else ""
    url = f"https://huggingface.co/{prefix}{repository}/tree/{revision}"
    print(f"Published and pinned: {url}")
    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with Path(summary).open("a") as stream:
            stream.write(f"\nHugging Face {kind}: [{revision}]({url})\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("github", "github-receipt", "model", "space"))
    parser.add_argument("folder", type=Path)
    parser.add_argument("repository")
    parser.add_argument("--tag", default="MAMBO_v3")
    parser.add_argument("--receipt", type=Path, help="Write the immutable Hub revision outside the payload directory")
    args = parser.parse_args()
    if args.kind == "github":
        github(args.folder.resolve(), args.repository, args.tag)
    elif args.kind == "github-receipt":
        github_files([args.folder.resolve()], args.repository, args.tag)
    else:
        if args.receipt and args.receipt.resolve().is_relative_to(args.folder.resolve()):
            parser.error("Receipt must be outside the publication payload")
        receipt = hub(args.folder.resolve(), args.repository, args.kind)
        if args.receipt:
            args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
