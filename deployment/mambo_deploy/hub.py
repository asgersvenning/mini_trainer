"""Hub transport only; bundle validation and runtime loading stay with Predictor."""

import json
from pathlib import Path

from .bundle import Bundle


def load_bundle(source, *, backend, embeddings, **options):
    local = Path(source).expanduser()
    if local.is_dir():
        return local / "bundle" if (local / "bundle/release.json").is_file() else local
    if backend not in ("torch", "onnx"):
        raise ValueError("backend must be 'torch' or 'onnx'")
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError as error:
        raise ImportError("Install mambo-v3[hub] for Hugging Face downloads") from error
    # hf_hub_download resolves branches once and handles cached/offline revisions.
    manifest_path = Path(hf_hub_download(str(source), "bundle/release.json", **options))
    revision = manifest_path.parent.parent.name
    manifest = json.loads(manifest_path.read_text())
    origins = manifest.get("origins", {})
    files = set(manifest["files"]) - set(origins)
    profile = "torch" if backend == "torch" else "onnx-embedding" if embeddings else "onnx"
    files.update(manifest["profiles"][profile]["files"])
    files.add("release.json")
    for name in files:
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError(f"Bundle path escapes root: {name}")
    root = snapshot_download(str(source), **{**options, "revision": revision}, allow_patterns=["bundle/" + name for name in sorted(files)])

    def fetch(relative):
        return hf_hub_download(str(source), "bundle/" + relative, **{**options, "revision": revision})

    return Bundle(Path(root) / "bundle", fetch=fetch)
