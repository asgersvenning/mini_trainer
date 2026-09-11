"""Content-addressed completed analysis; viewer assets are deliberately not cached."""

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

ANALYSIS_VERSION = 1


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def analysis_key(weights, *, angular, synthetic, threads):
    contract = {
        "version": ANALYSIS_VERSION,
        "checkpoint": file_hash(weights),
        "angular": angular,
        "synthetic": synthetic,
        "threads": threads,
    }
    return hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest(), contract


def read_analysis(root, key, contract):
    entry = Path(root) / key
    try:
        manifest = json.loads((entry / "manifest.json").read_text())
        if manifest["contract"] != contract:
            return None
        result = {}
        for name in ("report-data.json", "summary.json"):
            raw = (entry / name).read_bytes()
            if hashlib.sha256(raw).hexdigest() != manifest["files"][name]:
                return None
            json.loads(raw)
            result[name] = raw
        os.utime(entry, None)
        return result
    except (OSError, ValueError, KeyError, TypeError):
        return None


def store_analysis(root, key, contract, output, *, max_bytes=2 * 1024**3):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".pending-{os.getpid()}-", dir=root) as temporary:
        pending = Path(temporary)
        hashes = {}
        for name in ("report-data.json", "summary.json"):
            shutil.copyfile(Path(output) / name, pending / name)
            hashes[name] = file_hash(pending / name)
        (pending / "manifest.json").write_text(json.dumps({"contract": contract, "files": hashes}))
        target = root / key
        # Another process may already have published this complete key.
        if target.exists() and read_analysis(root, key, contract) is None:
            shutil.rmtree(target)
        try:
            pending.rename(target)
        except OSError:
            if read_analysis(root, key, contract) is None:
                raise
    entries = sorted((p for p in root.iterdir() if p.is_dir() and len(p.name) == 64), key=lambda p: p.stat().st_mtime, reverse=True)
    total = 0
    for entry in entries:
        total += sum(p.stat().st_size for p in entry.iterdir() if p.is_file())
        if total > max_bytes:
            shutil.rmtree(entry)


def clear_analysis_cache(root):
    """Remove only complete content-addressed entries owned by this cache."""
    root = Path(root)
    removed = 0
    if root.exists():
        for entry in root.iterdir():
            if entry.is_dir() and len(entry.name) == 64 and all(c in "0123456789abcdef" for c in entry.name):
                shutil.rmtree(entry)
                removed += 1
    return removed
