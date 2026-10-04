"""Create, verify and fetch explicit, portable research artifact manifests."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path, PurePosixPath
from urllib.parse import quote, urlsplit
from urllib.request import urlopen


def artifact_path(root, name):
    """Keep manifest members inside the chosen root, including through symlinks."""
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts or "\\" in name or str(path) != name:
        raise ValueError(f"Invalid artifact path: {name}")
    result = root / name
    if not result.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Artifact escapes root: {name}")
    return result


def fingerprint(path):
    with path.open("rb") as stream:
        return {"bytes": path.stat().st_size, "sha256": hashlib.file_digest(stream, "sha256").hexdigest()}


def create(root, names, revision):
    files = {name: fingerprint(artifact_path(root, name)) for name in sorted(set(names))}
    if not files:
        raise ValueError("An explicit nonempty artifact list is required")
    return {"schema": 1, "analysis_revision": revision, "files": files}


def restore(manifest, root, base_url=None):
    """Verify cached files; fetch missing/corrupt files atomically when requested."""
    if manifest.get("schema") != 1 or not manifest.get("files"):
        raise ValueError("Unsupported or empty artifact manifest")
    if base_url is not None and urlsplit(base_url).scheme != "https":
        raise ValueError("Downloads require an HTTPS base URL")
    for name, expected in manifest["files"].items():
        path = artifact_path(root, name)
        if path.is_file() and fingerprint(path) == expected:
            continue
        if base_url is None:
            raise ValueError(f"Missing or corrupt artifact: {name}")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = artifact_path(root, name + ".part")
        try:
            with urlopen(base_url.rstrip("/") + "/" + quote(name, safe="/"), timeout=120) as response:
                with temporary.open("wb") as stream:
                    shutil.copyfileobj(response, stream)
            if fingerprint(temporary) != expected:
                raise ValueError(f"Downloaded artifact checksum mismatch: {name}")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("create")
    make.add_argument("root", type=Path)
    make.add_argument("manifest", type=Path)
    make.add_argument("--files", type=Path, required=True, help="One relative file path per line; no directory recursion")
    make.add_argument("--revision", required=True, help="Pinned analysis source revision; training provenance stays in run artifacts")
    for name in ["verify", "fetch"]:
        command = commands.add_parser(name)
        command.add_argument("manifest", type=Path)
        command.add_argument("root", type=Path)
        if name == "fetch":
            command.add_argument("--base-url", required=True, help="Read-only HTTPS archive URL")
    args = parser.parse_args()
    if args.command == "create":
        result = create(args.root, args.files.read_text().splitlines(), args.revision)
        with args.manifest.open("x") as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")
    else:
        restore(json.loads(args.manifest.read_text()), args.root, getattr(args, "base_url", None))


if __name__ == "__main__":
    main()
