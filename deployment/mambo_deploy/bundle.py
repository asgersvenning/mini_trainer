"""Validated, relocatable bundle paths. Explicit local bundles stay offline; the default cache can fetch missing files."""

import hashlib
import json
import os
from pathlib import Path

from .download import cached_model_file, fetch_file, link_file


class Bundle:
    def __init__(self, root, *, download=False, fetch=None):
        self.download = download
        self.fetch = fetch
        self.root = Path(root).expanduser().resolve()
        manifest_bytes = (self.root / "release.json").read_bytes()
        self.manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        self.manifest = json.loads(manifest_bytes)
        if self.manifest.get("schema") != "mambo-release-v1":
            raise ValueError("Unsupported MAMBO bundle schema")
        if self.manifest.get("score_semantics") != "hierarchical-leaf-logits-logsumexp-v1":
            raise ValueError("Unsupported score semantics")
        self._verified = set()
        self.classes = self.read_json("classes.json")
        self.preprocessing = self.read_json("preprocessing.json")
        self.regions = self.read_json("presets.json")
        for rank, labels in enumerate(self.classes["labels"]):
            if not labels or len(set(labels)) != len(labels):
                raise ValueError(f"Invalid class vocabulary at rank {rank}")
        if len(self.classes["labels"]) != 3 or len(self.classes["parents"]) != 2:
            raise ValueError("This release requires species, genus and family ranks")
        for rank, parents in enumerate(self.classes["parents"]):
            if len(parents) != len(self.classes["labels"][rank]):
                raise ValueError("Parent mapping length mismatch")
            if any(not isinstance(p, int) or not 0 <= p < len(self.classes["labels"][rank + 1]) for p in parents):
                raise ValueError("Invalid parent index")

    def file(self, relative):
        candidate = self.root / relative
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError(f"Bundle path escapes root: {relative}")
        path = candidate.resolve()
        hub_blob = self.root.parent.parent.name == "snapshots" and any(
            path.is_relative_to(parent / "blobs") for parent in (self.root.parents[2], self.root.parents[3])
        )
        if not path.is_relative_to(self.root) and not hub_blob:
            raise ValueError(f"Bundle path escapes root: {relative}")
        item = self.manifest["files"].get(relative)
        if item is None:
            raise ValueError(f"Unlisted bundle file: {relative}")
        if relative not in self._verified:
            if not candidate.exists() and self.fetch is not None:
                self.fetch(relative)
                return self.file(relative)
            if not path.exists() and self.download and relative in self.manifest.get("origins", {}):
                cached_model_file(
                    self.manifest["origins"][relative],
                    path,
                    cache=self.root.parent,
                    **item,
                    offline=os.environ.get("MAMBO_OFFLINE") == "1",
                )
            if path.stat().st_size != item["size"]:
                raise ValueError(f"Bundle size mismatch: {relative}")
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != item["sha256"]:
                raise ValueError(f"Bundle hash mismatch: {relative}")
            self._verified.add(relative)
        return candidate

    def read_json(self, relative):
        return json.loads(self.file(relative).read_text())

    def profile(self, name):
        profile = self.manifest["profiles"][name]
        for relative in profile["files"]:
            self.file(relative)
        model = self.file(profile["model"])
        if name.startswith("onnx") and self.root.parent.parent.name == "snapshots":
            # ORT validates real paths of external weights. Hub symlinks can point
            # into different blob directories, so colocate verified bytes once.
            runtime = self.root.parents[3] / "mambo" / self.manifest_sha256
            for relative in profile["files"]:
                destination = runtime / relative
                if not destination.exists():
                    link_file(self.file(relative), destination)
                fetch_file("", destination, **self.manifest["files"][relative], offline=True)
            return runtime / profile["model"]
        return model
