"""Verify ordered, bounded overlap preserves the original pixels and TTA aggregation."""

import hashlib
import threading

import numpy as np
import pytest
from PIL import Image

from deployment.mambo_deploy.augmentation import infer_augmented, infer_prepared, resolve_tta
from deployment.mambo_deploy.preprocessing import preprocess
from dev.releases.mambo_v3 import prefetch


@pytest.mark.parametrize("recipe", ["none", "rotation30_pad25_3"])
def test_prepared_views_and_results_are_identical(tmp_path, recipe):
    records = []
    for i in range(3):
        path = tmp_path / f"{i}.png"
        Image.fromarray(np.random.default_rng(i).integers(0, 256, (31, 47, 3), dtype=np.uint8)).save(path)
        records.append({"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    paths = [tmp_path / r["path"] for r in records]
    tta = resolve_tta(recipe)

    def runtime(images, embeddings):
        values = images.mean(axis=(2, 3))
        return values, values.copy() if embeddings else None

    batches = list(prefetch.prepared_batches(records, tmp_path, 3, 2, 2, tta))
    views = batches[0][2]
    if tta is None:
        np.testing.assert_array_equal(views[0], np.stack([preprocess(path) for path in paths]))
    else:
        expected = infer_augmented(runtime, paths, tta, True)
        observed = infer_prepared(runtime, views, len(views), True)
        for a, b in zip(expected, observed, strict=True):
            np.testing.assert_array_equal(a, b)


def test_corrupt_image_fails_before_decode(tmp_path):
    (tmp_path / "bad").write_bytes(b"wrong bytes")
    records = [{"path": "bad", "sha256": "0" * 64}]
    with pytest.raises(ValueError, match="Image bytes changed"):
        list(prefetch.prepared_batches(records, tmp_path, 1, 2, 2))


def test_prefetch_is_bounded_and_advances_while_consumer_is_busy(monkeypatch, tmp_path):
    prepared = []
    third = threading.Event()

    def prepare(record, root, tta):
        prepared.append(record)
        if record == 2:
            third.set()
        return (np.full((1,), record),)

    monkeypatch.setattr(prefetch, "prepare_record", prepare)
    batches = prefetch.prepared_batches(list(range(20)), tmp_path, 1, 1, 2)
    assert next(batches)[0] == 0
    assert third.wait(timeout=2), "Producer should advance while consumer holds the first batch"
    assert prepared == [0, 1, 2]  # current batch + two ahead; not the entire request
    assert next(batches)[0] == 1
    batches.close()


def test_unprefetched_mode_remains_lazy(monkeypatch, tmp_path):
    prepared = []

    def prepare(record, root, tta):
        prepared.append(record)
        return (np.full((1,), record),)

    monkeypatch.setattr(prefetch, "prepare_record", prepare)
    batches = prefetch.prepared_batches(list(range(5)), tmp_path, 1, 0, 0)
    assert next(batches)[0] == 0
    assert prepared == [0]
    batches.close()
