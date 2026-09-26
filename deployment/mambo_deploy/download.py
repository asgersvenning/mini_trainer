"""Pinned ERDA downloads with atomic, verified caching and explicit offline support."""

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from urllib.request import urlopen


def fetch_file(url, destination, *, size, sha256, offline=False):
    destination = Path(destination)
    if destination.exists():
        with destination.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if destination.stat().st_size != size or digest != sha256:
            raise ValueError(f"Cached artifact integrity mismatch: {destination}")
        return destination
    if offline:
        raise FileNotFoundError(f"Artifact is not cached: {destination}. Disable offline mode to download it first.")
    if not url.startswith("https://"):
        raise ValueError("Artifact URL must use HTTPS")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".download-", delete=False) as stream:
            partial = Path(stream.name)
            digest = hashlib.sha256()
            total = 0
            with urlopen(url, timeout=60) as response:
                while chunk := response.read(1024 * 1024):
                    total += len(chunk)
                    if total > size:
                        raise ValueError(f"Downloaded artifact exceeds expected size: {url}")
                    stream.write(chunk)
                    digest.update(chunk)
            if total != size or digest.hexdigest() != sha256:
                raise ValueError(f"Downloaded artifact integrity mismatch: {url}")
        partial.replace(destination)
    finally:
        if partial is not None:
            partial.unlink(missing_ok=True)
    return destination


def cached_model_file(url, destination, *, cache, size, sha256, offline=False):
    """Reuse immutable weight bytes across metadata revisions, without symlinks."""
    destination = Path(destination)
    blob = Path(cache) / "blobs" / sha256
    fetch_file(url, blob, size=size, sha256=sha256, offline=offline)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".model-", dir=destination.parent) as temporary:
        staged = Path(temporary) / "data"
        try:
            os.link(blob, staged)
        except OSError:
            shutil.copyfile(blob, staged)
        staged.replace(destination)


def default_bundle():
    """Install small packaged metadata; model files are fetched lazily by Bundle."""
    descriptor = json.loads(Path(__file__).with_name("default_bundle.json").read_text())
    revision = hashlib.sha256(json.dumps(descriptor, sort_keys=True).encode()).hexdigest()[:16]
    cache = Path(os.environ.get("MAMBO_CACHE", Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "mambo"))
    root = cache.expanduser() / revision
    for relative, content in descriptor["metadata"].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("Packaged metadata path escapes cache")
        data = content.encode()
        if path.exists():
            if path.read_bytes() != data:
                raise ValueError(f"Cached metadata differs from this release: {path}")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temp = Path(stream.name)
            stream.write(data)
        temp.replace(path)
    return root
