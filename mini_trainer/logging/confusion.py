"""Whole-matrix confusion diagnostics: bounded reduction, images and exact data.

Dashboard overviews and full-resolution local artifacts share a fixed log scale.
No class filtering or reordering occurs here. Model output indices identify axes.
"""

import json
import math
from pathlib import Path
from typing import NamedTuple

import matplotlib as mpl
import numpy as np
import torch
import torch.distributed as dist
from matplotlib.colors import LogNorm
from PIL import Image, ImageDraw

from mini_trainer.visualization.plot import _generate_colorbar_rgb_array, _get_colorbar_ticks_and_labels

PREVIEW_SIZE = 1536
PREVIEW_HEADER_HEIGHT = 24
PROBABILITY_MIN = 1e-6
SOFT_COLOR_LEVELS = 128
CHUNK_ELEMENTS = 262144
# Dense N×N buffers (soft sums, hard counts, full-resolution image) are bounded by this many cells,
# i.e. 16,384 classes. Larger heads keep exact sparse counts and the binned preview only.
MAX_DENSE_CELLS = 2**28


class PairCounts(NamedTuple):
    """Exact sparse confusion counts in row-major order; memory scales with observed pairs, not classes²."""

    rows: np.ndarray
    columns: np.ndarray
    counts: np.ndarray
    n: int


def reduce_matrix(matrix, *, copy=True):
    """Sum rank-local CPU matrices onto rank zero without a full GPU copy.

    Each call is collective; all ranks must supply the same shape and dtype.
    Soft buffers are copied so repeated reporting cannot double counts. Fresh
    hard-count buffers may pass copy=False to avoid another dense allocation.
    """
    if not dist.is_available() or not dist.is_initialized():
        return matrix
    device = torch.device("cuda", torch.cuda.current_device()) if dist.get_backend() == "nccl" else torch.device("cpu")
    result = (np.empty_like(matrix) if copy else matrix) if dist.get_rank() == 0 else None
    rows = max(1, CHUNK_ELEMENTS // max(1, matrix.shape[1]))
    for start in range(0, matrix.shape[0], rows):
        chunk = torch.tensor(matrix[start : start + rows], device=device)
        dist.reduce(chunk, dst=0)
        if result is not None:
            result[start : start + rows] = chunk.cpu().numpy()
    return result


def hard_counts(labels, predictions, n_classes):
    """Count observed (label, prediction) pairs; the model's output width fixes the class axes."""
    labels, predictions = np.asarray(labels, dtype=np.int64), np.asarray(predictions, dtype=np.int64)
    if labels.shape != predictions.shape:
        raise ValueError("Labels and predictions must have the same shape")
    if labels.size and (min(labels.min(), predictions.min()) < 0 or max(labels.max(), predictions.max()) >= n_classes):
        raise ValueError("Confusion class index outside model output range")
    codes, counts = np.unique(n_classes * labels + predictions, return_counts=True)
    return PairCounts(codes // n_classes, codes % n_classes, counts.astype(np.int64), n_classes)


def reduce_pairs(pairs: PairCounts):
    """Merge rank-local pair counts onto rank zero (other ranks get None). Each call is collective."""
    if not dist.is_available() or not dist.is_initialized():
        return pairs
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, pairs)
    if dist.get_rank() != 0:
        return None
    codes = np.concatenate([p.rows * pairs.n + p.columns for p in gathered])
    counts = np.concatenate([p.counts for p in gathered])
    unique, inverse = np.unique(codes, return_inverse=True)
    merged = np.bincount(inverse.ravel(), weights=counts, minlength=len(unique)).astype(np.int64)
    return PairCounts(unique // pairs.n, unique % pairs.n, merged, pairs.n)


def _palette(soft):
    cmap = mpl.colormaps["magma"].resampled(SOFT_COLOR_LEVELS) if soft else mpl.colormaps["magma"].copy()
    cmap.set_bad("magenta")
    cmap.set_under("black")
    palette = np.zeros((256, 3), dtype=np.uint8)
    palette[1 : SOFT_COLOR_LEVELS + 1] = cmap(np.linspace(0, 1, SOFT_COLOR_LEVELS), bytes=True)[:, :3]
    palette[SOFT_COLOR_LEVELS + 1] = (255, 0, 255)
    return cmap, palette


def _colors(values, soft, cmap):
    """Positive probabilities below the floor stay distinct from exact zero."""
    invalid = ~np.isfinite(values) | (values < 0)
    zero = values == 0
    log_values = np.log10(np.clip(np.where(invalid, 0, values), PROBABILITY_MIN, 1))
    normalized = (log_values - math.log10(PROBABILITY_MIN)) / -math.log10(PROBABILITY_MIN)
    if soft:
        indices = np.minimum((normalized * SOFT_COLOR_LEVELS).astype(np.uint8), SOFT_COLOR_LEVELS - 1) + 1
        indices[zero] = 0
        indices[invalid] = SOFT_COLOR_LEVELS + 1
        return indices
    rgb = cmap(normalized, bytes=True)[:, :, :3].copy()
    rgb[zero] = 0
    rgb[invalid] = (255, 0, 255)
    return rgb


def confusion_report(matrix, *, soft, directory=None):
    """Return a bounded whole-matrix preview; optionally save native-resolution data.

    Large previews show the arithmetic mean of row-normalized cell probabilities
    in each block. They are overviews, not aggregated/renormalized class matrices.
    Exact hard counts are sparse NPZ; soft sums are uncompressed NPY to avoid
    expensive compression of high-entropy floats during training.

    Hard counts may be passed as ``PairCounts``. Up to ``MAX_DENSE_CELLS`` cells they are
    densified and reported exactly as a dense matrix; beyond it only the exact sparse
    counts, support and block preview are produced, with no full-resolution image.
    """
    if isinstance(matrix, PairCounts):
        if matrix.n * matrix.n > MAX_DENSE_CELLS:
            return _sparse_report(matrix, directory)
        dense = np.zeros((matrix.n, matrix.n), dtype=np.int64)
        dense[matrix.rows, matrix.columns] = matrix.counts
        matrix = dense
    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or not matrix.shape[0] or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Confusion report requires a nonempty square matrix")
    n = len(matrix)
    support = matrix.sum(axis=1, dtype=np.float64 if soft else np.int64)
    block = max(1, math.ceil(n / PREVIEW_SIZE))
    edges = np.arange(0, n, block)
    widths = np.minimum(block, n - edges)
    preview = np.zeros((len(edges), len(edges)), dtype=np.float64)
    cmap, palette = _palette(soft)
    full = None
    if directory is not None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        full = np.empty((n, n) if soft else (n, n, 3), dtype=np.uint8)
        if soft:
            np.save(directory / "probability_sums.npy", matrix, allow_pickle=False)
        else:
            rows, columns = np.nonzero(matrix)
            np.savez_compressed(directory / "counts.npz", rows=rows, columns=columns, counts=matrix[rows, columns], shape=matrix.shape)
        np.save(directory / "row_support.npy", support, allow_pickle=False)
    rows_per_chunk = max(block, (CHUNK_ELEMENTS // max(1, n) // block) * block)
    invalid_cells = 0
    for start in range(0, n, rows_per_chunk):
        raw = matrix[start : start + rows_per_chunk]
        denominator = support[start : start + len(raw), None]
        values = np.zeros(raw.shape, dtype=np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            np.divide(raw, denominator, out=values, where=denominator != 0)
        values[(raw < 0) | ~np.isfinite(raw)] = np.nan
        invalid_cells += int(np.count_nonzero(~np.isfinite(values) | (values < 0)))
        if full is not None:
            full[start : start + len(raw)] = _colors(values, soft, cmap)
        sums = np.add.reduceat(np.add.reduceat(values, edges, axis=1), np.arange(0, len(raw), block), axis=0)
        lengths = np.minimum(block, len(raw) - np.arange(0, len(raw), block))
        preview[start // block : start // block + len(sums)] = sums / lengths[:, None] / widths[None, :]

    def image(values):
        if soft:
            result = Image.fromarray(values)
            result.putpalette(palette.ravel().tolist())
            return result
        return Image.fromarray(values)

    if directory is not None:
        image(full).save(directory / "matrix.png", compress_level=3)
        del full
        _write_metadata(directory, n, soft, block, cmap, invalid_cells)
    return _captioned_preview(image(_colors(preview, soft, cmap)).convert("RGB"), n, block, cmap, directory)


def _write_metadata(directory, n, soft, block, cmap, invalid_cells, **extra):
    metadata = {
        "shape": [n, n],
        "class_indices": list(range(n)),
        "axes": "rows=true class; columns=predicted class; original model output index order",
        "class_names": "Resolve indices using the saved model/config classifier metadata",
        "normalization": "divide each raw row by its sum; empty rows remain zero",
        "raw_values": "probability_sums.npy" if soft else "counts.npz",
        "support": "row_support.npy",
        "scope": "sum over reporting ranks (including any validation sampler padding)",
        "probability_scale": [PROBABILITY_MIN, 1],
        "positive_color_levels": SOFT_COLOR_LEVELS if soft else cmap.N,
        "zero_color": "black",
        "invalid_color": "magenta",
        "invalid_normalized_cells": invalid_cells,
        "preview": {"method": "arithmetic mean of cell probabilities", "block_shape": [block, block]},
        **extra,
    }
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def _captioned_preview(preview_image, n, block, cmap, directory):
    # Nearest-neighbor enlargement for small matrices only; never smooth cells.
    if preview_image.width < 500:
        factor = math.ceil(500 / preview_image.width)
        preview_image = preview_image.resize((preview_image.width * factor, preview_image.height * factor), Image.Resampling.NEAREST)
    norm = LogNorm(PROBABILITY_MIN, 1)
    ticks, labels = _get_colorbar_ticks_and_labels(PROBABILITY_MIN, 1, 8, True)
    colorbar = _generate_colorbar_rgb_array(norm, cmap, ticks, labels, preview_image.height, 10)
    if directory is not None:
        Image.fromarray(colorbar).save(directory / "colorbar.png", compress_level=3)
    combined = Image.fromarray(np.hstack((np.asarray(preview_image), colorbar)))
    captioned = Image.new("RGB", (combined.width, combined.height + PREVIEW_HEADER_HEIGHT), "white")
    captioned.paste(combined, (0, PREVIEW_HEADER_HEIGHT))
    ImageDraw.Draw(captioned).text(
        (8, 4),
        f"{n} classes | rows=true, columns=predicted | block mean {block}x{block} | black=0; magenta=invalid",
        fill="black",
    )
    return np.asarray(captioned)


def _sparse_report(pairs: PairCounts, directory):
    """Hard-count report whose memory scales with observed pairs: exact counts, support and block preview only."""
    n = pairs.n
    support = np.bincount(pairs.rows, weights=pairs.counts, minlength=n).astype(np.int64)
    block = max(1, math.ceil(n / PREVIEW_SIZE))
    edges = np.arange(0, n, block)
    size = len(edges)
    values = pairs.counts / support[pairs.rows]
    cell = (pairs.rows // block) * size + pairs.columns // block
    widths = np.minimum(block, n - edges)
    preview = np.bincount(cell, weights=values, minlength=size * size).reshape(size, size) / widths[:, None] / widths[None, :]
    cmap, _ = _palette(False)
    if directory is not None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(directory / "counts.npz", rows=pairs.rows, columns=pairs.columns, counts=pairs.counts, shape=(n, n))
        np.save(directory / "row_support.npy", support, allow_pickle=False)
        _write_metadata(
            directory, n, False, block, cmap, 0, full_resolution=f"omitted: {n}x{n} cells exceed the dense budget of {MAX_DENSE_CELLS}"
        )
    return _captioned_preview(Image.fromarray(_colors(preview, False, cmap)).convert("RGB"), n, block, cmap, directory)
