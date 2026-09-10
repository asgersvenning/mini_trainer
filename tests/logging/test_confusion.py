"""Whole-matrix integrity, bounded previews and distributed collection."""

import json
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from PIL import Image

from mini_trainer.logging import confusion as report
from mini_trainer.logging.core import MultiLogger


def test_hard_report_retains_unobserved_classes_and_counts(tmp_path):
    logger = MultiLogger([0], [0], 1, str(tmp_path), "run")
    logger.update(epoch=0, type="eval")
    logger._n_classes = 4
    logger.log_labels_predictions([0, 0, 1], [0, 2, 1])
    figures = logger.confusion_matrix()
    assert list(figures) == ["Confusion matrix/lvl0/overview_mean"]
    root = tmp_path / "run/logs/figures/epoch-0001/Confusion_matrix_lvl0"
    data = np.load(root / "counts.npz")
    counts = np.zeros(tuple(data["shape"]), dtype=np.int64)
    counts[data["rows"], data["columns"]] = data["counts"]
    assert counts.shape == (4, 4)
    assert counts[0, 2] == 1
    assert counts.sum() == 3
    np.testing.assert_array_equal(np.load(root / "row_support.npy"), [2, 1, 0, 0])
    assert Image.open(root / "matrix.png").size == (4, 4)
    assert json.loads((root / "metadata.json").read_text())["class_indices"] == [0, 1, 2, 3]


def test_soft_palette_zeros_raw_values_and_mean_overview(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "PREVIEW_SIZE", 2)
    values = np.array([[0, 0.2, 0.8, 0, 0], [0, 0, 0, 0, 0], [1e-9, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]], dtype=np.float32)
    original = values.copy()
    preview = report.confusion_report(values, soft=True, directory=tmp_path)
    np.testing.assert_array_equal(values, original)
    np.testing.assert_array_equal(np.load(tmp_path / "probability_sums.npy"), original)
    image = Image.open(tmp_path / "matrix.png")
    assert image.mode == "P"
    indices = np.asarray(image)
    assert np.all(indices[values == 0] == 0)
    assert indices[2, 0] != 0  # Below the positive floor still differs from zero.
    assert indices.max() <= 128
    cmap, palette = report._palette(True)
    support = values.sum(axis=1, dtype=np.float64)
    normalized = np.divide(values, support[:, None], out=np.zeros(values.shape), where=support[:, None] != 0)
    expected = np.array([[normalized[:3, :3].mean(), normalized[:3, 3:].mean()], [normalized[3:, :3].mean(), normalized[3:, 3:].mean()]])
    expected_rgb = palette[report._colors(expected, True, cmap)]
    np.testing.assert_array_equal(preview[report.PREVIEW_HEADER_HEIGHT : report.PREVIEW_HEADER_HEIGHT + 500 : 250, :500:250], expected_rgb)
    np.testing.assert_array_equal(np.asarray(image.convert("RGB")), palette[indices])


def test_invalid_soft_cells_are_distinct_from_zero(tmp_path):
    values = np.array([[np.nan, 1], [0, 0]], dtype=np.float32)
    report.confusion_report(values, soft=True, directory=tmp_path)
    image = np.asarray(Image.open(tmp_path / "matrix.png").convert("RGB"))
    np.testing.assert_array_equal(image[0, 0], [255, 0, 255])
    np.testing.assert_array_equal(image[1, 0], [0, 0, 0])
    assert json.loads((tmp_path / "metadata.json").read_text())["invalid_normalized_cells"] == 2


def _ddp_confusion_worker(rank, rendezvous, root):
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    try:
        report.CHUNK_ELEMENTS = 4  # Exercise multiple collective chunks.
        logger = MultiLogger([0], [0], 1, root, "ddp")
        logger.update(epoch=0, type="eval")
        # Rank one alone has observations; rank zero must discover its shape.
        if rank == 1:
            logger._n_classes = 3
            logger.log_labels_predictions([0, 1], [2, 1])
            logger._soft_confusion_matrix[0] = torch.tensor([[0.1, 0.2, 0.7], [0, 1.0, 0], [0, 0, 0]])
        original = {k: v.clone() for k, v in logger._soft_confusion_matrix.items()}
        for _ in range(2):
            figures = logger.confusion_matrix()
            assert bool(figures) == (rank == 0)
        for key in original:
            torch.testing.assert_close(logger._soft_confusion_matrix[key], original[key])
    finally:
        dist.destroy_process_group()


def test_ddp_confusions_include_other_rank_and_do_not_double_count(tmp_path):
    mp.spawn(_ddp_confusion_worker, args=(str(tmp_path / "rendezvous"), str(tmp_path)), nprocs=2, join=True)
    root = Path(tmp_path) / "ddp/logs/figures/epoch-0001"
    with np.load(root / "Confusion_matrix_lvl0/counts.npz") as saved:
        assert saved["counts"].sum() == 2
        assert list(zip(saved["rows"], saved["columns"], strict=True)) == [(0, 2), (1, 1)]
    np.testing.assert_allclose(np.load(root / "Soft_confusion_matrix_lvl0/probability_sums.npy"), [[0.1, 0.2, 0.7], [0, 1, 0], [0, 0, 0]])
