"""Predict with a mini_trainer release: image -> trunk (ONNX) -> head (safetensors) -> every rank.

This file is copied into each release as ``predict.py`` and imports nothing from mini_trainer, so a
release runs with ``pip install onnxruntime numpy pillow safetensors``:

    python predict.py photo.jpg more.jpg --top 3            # the default head
    python predict.py photo.jpg --head moths --trunk trunk_fp16.onnx

    from predict import Release
    release = Release("path/to/release")
    results = release.predict(["photo.jpg"], top=3)        # per image: {rank: [(label, probability), ...]}
    features, embedding = release.embed(["photo.jpg"])      # embedding: L2-normalised, cosine = dot product

A release splits the model in two. The **trunk** maps a square uint8 image to ``features`` (what the
classifier reads) and ``embedding`` (the default head's normalised embedding). A **head** is a safetensors
file: an optional hidden layer, one unit prototype per class with a learned magnitude ``g`` and bias
``b``, and the parent of every class at each rank. :func:`score` reproduces mini_trainer's normalised
head exactly: ``logit = sqrt(D - 2) * asin(clamp(g * cos)) + b`` (or ``g * sqrt(D - 2) * asin(clamp(cos)) + b``
when the head was trained with ``magnitude_after_angle``), and each coarser rank is the logsumexp of
its children, so ``softmax`` of any rank is that rank's marginal probability.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

EPS = 1e-7  # mini_trainer's cosine clamp: cos in [-1 + EPS, 1 - EPS]


# --------------------------------------------------------------------------------------- preprocessing
def read_square(path: str | Path, size: int) -> np.ndarray:
    """The largest centred square of an image, resized to ``size`` with PIL bilinear; uint8 ``[3, size, size]``.

    Identical to mini_trainer's evaluation reader (``read_square_crop`` at the centre): the same crop box,
    including its rounding, and the same PIL filter. EXIF orientation is not applied. The rest of the
    preprocessing (resampling to the model size and normalisation) is inside the trunk graph.
    """
    with Image.open(path) as image:
        image = image.convert("RGB")
    width, height = image.size
    side = int((width / size if width / size < height / size else height / size) * size)
    left, top = int(0.5 * (width - side)), int(0.5 * (height - side))
    image = image.crop((left, top, left + side, top + side)).resize((size, size), Image.Resampling.BILINEAR)
    return np.ascontiguousarray(np.asarray(image).transpose(2, 0, 1))


# ------------------------------------------------------------------------------------------------- head
@dataclass
class Head:
    """One classification head: optional hidden layer, unit prototypes, magnitudes, biases and the tree."""

    direction: np.ndarray  # [C, D] unit rows
    magnitude: np.ndarray  # [C] learned row norm g
    bias: np.ndarray  # [C]
    parents: list[np.ndarray]  # parents[k][i]: index at rank k + 1 of node i at rank k
    ranks: list[str]
    labels: list[list[str]] | None = None  # labels[k][i]: name of node i at rank k
    hidden_weight: np.ndarray | None = None  # [D, F]
    hidden_bias: np.ndarray | None = None
    magnitude_after_angle: bool = False
    sizes: list[int] = field(default_factory=list)  # nodes per rank
    hidden_activation: str = "leaky_relu"  # or "none": a linear hidden layer

    def __post_init__(self):
        if not self.sizes:
            self.sizes = [len(self.direction)] + [int(p.max()) + 1 for p in self.parents]


def load_head(path: str | Path, labels: str | Path | None = None, prefix: str = "") -> Head:
    """Read a head written by ``mt_release`` (and its labels JSON, when given).

    ``prefix="head."`` reads the head out of the combined ``model.safetensors``.
    """
    from safetensors import safe_open

    with safe_open(str(path), framework="numpy") as handle:
        meta = json.loads(handle.metadata()["mini_trainer_head"])
        tensors = {key.removeprefix(prefix): handle.get_tensor(key) for key in handle.keys() if key.startswith(prefix)}
    names = json.loads(Path(labels).read_text())["labels"] if labels else None
    return Head(
        direction=tensors["direction"],
        magnitude=tensors["magnitude"],
        bias=tensors["bias"],
        parents=[tensors[f"parent.{k}"] for k in range(len(meta["ranks"]) - 1)],
        ranks=meta["ranks"],
        labels=names,
        hidden_weight=tensors.get("hidden.weight"),
        hidden_bias=tensors.get("hidden.bias"),
        magnitude_after_angle=meta["magnitude_after_angle"],
        sizes=meta["sizes"],
        hidden_activation=meta.get("hidden_activation", "leaky_relu"),
    )


def subset(head: Head, rows) -> Head:
    """The head restricted to some classes of its first rank (a species list), as mini_trainer's class masks do.

    Coarser ranks keep only the nodes that still have a class below them; ``labels`` follow, so outputs
    stay named. ``rows`` are indices into the first rank (sorted on the way in).
    """
    keep = [np.unique(np.asarray(rows, dtype=np.int64))]
    parents = []
    for parent in head.parents:
        nodes, inverse = np.unique(parent[keep[-1]], return_inverse=True)
        parents.append(inverse.astype(np.int64))
        keep.append(nodes)
    rows = keep[0]
    return Head(
        direction=head.direction[rows],
        magnitude=head.magnitude[rows],
        bias=head.bias[rows],
        parents=parents,
        ranks=head.ranks,
        labels=None if head.labels is None else [[names[i] for i in idx] for names, idx in zip(head.labels, keep)],
        hidden_weight=head.hidden_weight,
        hidden_bias=head.hidden_bias,
        magnitude_after_angle=head.magnitude_after_angle,
        hidden_activation=head.hidden_activation,
    )


def zscore(cosine: np.ndarray, dim: int) -> np.ndarray:
    """mini_trainer's ``cosine_to_zscore``: ``sqrt(D - 2) * asin(clamp(cosine))``, written as it is there."""
    return (np.arccos(-np.clip(cosine, -1 + EPS, 1 - EPS)) - np.pi / 2) / (1 / (dim - 2) ** 0.5)


def logsumexp_by_parent(logits: np.ndarray, parent: np.ndarray, size: int) -> np.ndarray:
    """``out[:, j] = logsumexp(logits[:, parent == j])``; ``-inf`` for a node without children."""
    order = np.argsort(parent, kind="stable")
    grouped = parent[order]
    starts = np.flatnonzero(np.r_[True, grouped[1:] != grouped[:-1]])
    values = logits[:, order]
    peak = np.maximum.reduceat(values, starts, axis=1)
    total = np.add.reduceat(np.exp(values - np.repeat(peak, np.diff(np.r_[starts, len(grouped)]), axis=1)), starts, axis=1)
    out = np.full((len(logits), size), -np.inf, dtype=logits.dtype)
    out[:, grouped[starts]] = np.log(total) + peak
    return out


def embed_features(features: np.ndarray, head: Head) -> np.ndarray:
    """The head's embedding: hidden layer and its activation (leaky ReLU unless "none") when it has one, then L2
    normalisation."""
    x = features.astype(np.float32)
    if head.hidden_weight is not None:
        x = x @ head.hidden_weight.T + head.hidden_bias
        if head.hidden_activation == "leaky_relu":
            x = np.where(x >= 0, x, np.float32(0.01) * x)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), np.float32(1e-12))


def score(features: np.ndarray, head: Head) -> tuple[np.ndarray, list[np.ndarray]]:
    """Trunk features -> (embedding, logits per rank, finest first)."""
    embedding = embed_features(features, head)
    cosine = embedding @ head.direction.T
    dim = head.direction.shape[1]
    if head.magnitude_after_angle:
        logits = zscore(cosine, dim) * head.magnitude + head.bias
    else:
        logits = zscore(cosine * head.magnitude, dim) + head.bias
    ranks = [logits]
    for parent, size in zip(head.parents, head.sizes[1:]):
        ranks.append(logsumexp_by_parent(ranks[-1], parent, size))
    return embedding, ranks


def softmax(logits: np.ndarray) -> np.ndarray:
    z = np.exp(logits - logits.max(axis=1, keepdims=True))
    return z / z.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------------------------- release
def onnx_session(path: str | Path, threads: int | None = None):
    """An onnxruntime session on CUDA when available, else CPU."""
    import onnxruntime as ort

    options = ort.SessionOptions()
    if threads:
        options.intra_op_num_threads = threads
    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available and hasattr(ort, "preload_dlls"):
        ort.preload_dlls()  # pip-installed CUDA libraries load only this way; without it CUDA falls back to CPU
    # Not get_available_providers() as is: it can list TensorRT first and fall back to CPU silently.
    providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]
    return ort.InferenceSession(str(path), options, providers=providers)


class Release:
    """A release folder: ``config.json``, a trunk graph and one or more heads."""

    def __init__(self, folder: str | Path, head: str | None = None, trunk: str | None = None, threads: int | None = None):
        self.folder = Path(folder)
        self.config = json.loads((self.folder / "config.json").read_text())
        self.read_size = self.config["preprocessing"]["read_size"]
        self.session = onnx_session(self.folder / (trunk or self.config["trunk"]["onnx"]), threads)
        spec = self.config["heads"][head or self.config["default_head"]]
        self.head = load_head(self.folder / spec["file"], self.folder / spec["labels"])

    def pixels(self, paths) -> np.ndarray:
        return np.stack([read_square(path, self.read_size) for path in paths])

    def embed(self, paths) -> tuple[np.ndarray, np.ndarray]:
        """(features, embedding) from the trunk graph."""
        features, embedding = self.session.run(["features", "embedding"], {"images": self.pixels(paths)})
        return features, embedding

    def predict(self, paths, top: int = 1, head: Head | None = None) -> list[dict]:
        """Top-``top`` (label, probability) per rank and image; ``head`` overrides (e.g. a :func:`subset`)."""
        head = head or self.head
        features, _ = self.embed(paths)
        _, ranks = score(features, head)
        results = [{} for _ in paths]
        for rank, name, labels in zip(ranks, head.ranks, head.labels or [None] * len(head.ranks)):
            probabilities = softmax(rank)
            best = np.argsort(-probabilities, axis=1)[:, :top]
            for result, row, index in zip(results, probabilities, best):
                result[name] = [(labels[i] if labels else int(i), float(row[i])) for i in index]
        return results


def torch_trunk(folder: str | Path, combined: bool = False):
    """The trunk as a PyTorch module from ``trunk.safetensors`` (needs torch and timm; timm backbones only).

    ``combined=True`` reads it from ``model.safetensors`` instead.

    Takes the same uint8 square images as the ONNX trunk and returns ``features``: the resample to the
    model size (identity grid, bilinear, reflection padding, ``align_corners=True``, as mini_trainer's
    evaluation transform) and the normalisation are those of the graph.
    """
    import timm
    import torch
    import torch.nn.functional as F
    from safetensors.torch import load_file

    folder = Path(folder)
    config = json.loads((folder / "config.json").read_text())
    backbone, prep = config["trunk"]["backbone"], config["preprocessing"]
    if not backbone.startswith("timm:"):
        raise ValueError(f"torch_trunk rebuilds timm backbones only, not {backbone}")
    if prep["read_size"] > 2 * prep["model_size"]:  # mini_trainer then area-downsamples first; not reproduced here
        raise ValueError("torch_trunk supports read sizes up to twice the model size")
    model = timm.create_model(backbone.removeprefix("timm:"), pretrained=False, num_classes=0)
    if combined:
        from safetensors import safe_open

        with safe_open(str(folder / "model.safetensors"), framework="pt") as handle:
            state = {k.removeprefix("trunk."): handle.get_tensor(k) for k in handle.keys() if k.startswith("trunk.")}
    else:
        state = load_file(str(folder / config["trunk"]["safetensors"]))
    model.load_state_dict(state, strict=True)
    mean = torch.tensor(prep["mean"]).view(1, 3, 1, 1)
    std = torch.tensor(prep["std"]).view(1, 3, 1, 1)

    class Trunk(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = model.eval()

        def forward(self, images):
            x = images.float().div(255)
            size = prep["model_size"]
            if x.shape[-1] != size:
                theta = torch.eye(2, 3).expand(len(x), 2, 3)
                grid = F.affine_grid(theta, [len(x), 3, size, size], align_corners=True)
                x = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=True)
            return self.backbone((x - mean) / std)

    return Trunk().eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images", nargs="+")
    parser.add_argument("--release", default=str(Path(__file__).resolve().parent), help="release folder (default: this file's)")
    parser.add_argument("--head", default=None, help="head name in config.json (default: its default_head)")
    parser.add_argument("--trunk", default=None, help="trunk graph, e.g. trunk_fp16.onnx (default: config)")
    parser.add_argument("--top", type=int, default=1)
    args = parser.parse_args()
    release = Release(args.release, head=args.head, trunk=args.trunk)
    for path, result in zip(args.images, release.predict(args.images, top=args.top)):
        print(path)
        for rank, candidates in result.items():
            print(f"  {rank:10s} " + ", ".join(f"{label} {p:.3f}" for label, p in candidates))


if __name__ == "__main__":
    main()
