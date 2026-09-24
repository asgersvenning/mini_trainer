import hashlib
import threading
from contextlib import closing

import numpy as np
import pytest
from PIL import Image

from deployment.mambo_deploy import streaming
from deployment.mambo_deploy.augmentation import resolve_tta
from deployment.mambo_deploy.preprocessing import preprocess


def inputs(tmp_path, count=9):
    items = []
    for i in range(count):
        path = tmp_path / f"{i}.png"
        Image.fromarray(np.full((20, 30, 3), i, dtype=np.uint8)).save(path)
        items.append((path, hashlib.sha256(path.read_bytes()).hexdigest()))
    return items


@pytest.mark.parametrize("tta", [None, resolve_tta("rotation30_pad25_3")])
def test_order_pixels_and_bounds(tmp_path, tta):
    items = inputs(tmp_path)
    stats = {}
    batches = list(
        streaming.prepared_stream(
            items, 2, tta=tta, read_workers=8, prepare_workers=3, read_window=8, prefetch_batches=2, encoded_budget=400, stats=stats
        )
    )
    assert [offset for offset, _ in batches] == [0, 2, 4, 6, 8]
    for offset, views in batches:
        expected = [streaming.prepare_image(p.read_bytes(), tta) for p, _ in items[offset : offset + len(views[0])]]
        for i, view in enumerate(views):
            np.testing.assert_array_equal(view, np.stack([image[i] for image in expected]))
    assert stats["peak_encoded_bytes"] <= 400
    assert stats["peak_prepared_images"] <= 6


def test_slow_first_read_does_not_block_later_preparation(tmp_path, monkeypatch):
    items = inputs(tmp_path)
    later = threading.Event()
    original_read, original_prepare = streaming.read_image, streaming.prepare_image

    def read(path, size, digest):
        if path == items[0][0]:
            assert later.wait(3), "later images must prepare while first read waits"
        return original_read(path, size, digest)

    def prepare(data, tta):
        result = original_prepare(data, tta)
        later.set()
        return result

    monkeypatch.setattr(streaming, "read_image", read)
    monkeypatch.setattr(streaming, "prepare_image", prepare)
    assert len(list(streaming.prepared_stream(items, 2, read_window=8))) == 5


def test_failure_close_and_empty(tmp_path):
    items = inputs(tmp_path)
    with pytest.raises(ValueError, match="bytes changed"):
        list(streaming.prepared_stream([(items[0][0], "bad")], 2))
    with pytest.raises(ValueError, match="budget"):
        list(streaming.prepared_stream(items, 2, encoded_budget=1))
    assert list(streaming.prepared_stream([], 2)) == []
    with closing(streaming.prepared_stream(items, 2)) as stream:
        offset, views = next(stream)
        assert offset == 0
        np.testing.assert_array_equal(views[0][0], preprocess(items[0][0]))
    assert not any(t.name == "mambo-stream" for t in threading.enumerate())


def test_assembly_runs_in_background_and_errors_propagate(tmp_path, monkeypatch):
    items = inputs(tmp_path, 5)
    original = streaming.assemble_batch
    calls = []

    def assemble(images):
        calls.append(threading.current_thread().name)
        result = original(images)
        assert images == []  # Per-image buffers are released by the assembler.
        return result

    monkeypatch.setattr(streaming, "assemble_batch", assemble)
    stats = {}
    assert len(list(streaming.prepared_stream(items, 2, stats=stats))) == 3
    assert len(calls) == 3 and all(name.startswith("mambo-assemble") for name in calls)
    assert stats["batch_assembly_seconds"] > 0
    assert stats["queue_wait_seconds"] == stats["input_wait_seconds"]

    def fail(images):
        raise ValueError("assembly failure")

    monkeypatch.setattr(streaming, "assemble_batch", fail)
    with pytest.raises(ValueError, match="assembly failure"):
        list(streaming.prepared_stream(items, 2))
    assert not any(t.name.startswith("mambo-assemble") for t in threading.enumerate())
