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

    def prepare(data, tta, out=None, **kwargs):
        result = original_prepare(data, tta, out=out, **kwargs)
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


def test_workers_fill_batch_storage_and_errors_propagate(tmp_path, monkeypatch):
    items = inputs(tmp_path, 5)
    original = streaming.prepare_image
    targets = []

    def prepare(data, tta, out=None, **kwargs):
        assert out is not None
        targets.append((threading.current_thread().name, out[0]))
        return original(data, tta, out=out, **kwargs)

    monkeypatch.setattr(streaming, "prepare_image", prepare)
    stats = {}
    batches = list(streaming.prepared_stream(items, 2, stats=stats))
    assert len(targets) == 5
    assert all(name.startswith("mambo-prepare") for name, _ in targets)
    assert all(any(np.shares_memory(target, views[0]) for _, views in batches) for _, target in targets)
    assert stats["preparation_worker_seconds"] > 0
    assert stats["queue_wait_seconds"] == stats["input_wait_seconds"]

    def fail(data, tta, out=None, **kwargs):
        raise ValueError("preparation failure")

    monkeypatch.setattr(streaming, "prepare_image", fail)
    with pytest.raises(ValueError, match="preparation failure"):
        list(streaming.prepared_stream(items, 2))
    assert not any(t.name.startswith("mambo-prepare") for t in threading.enumerate())


def test_result_worker_order_bounds_overlap_and_failure():
    from deployment.mambo_deploy.result_worker import ResultWorker

    release = threading.Event()
    started = threading.Event()

    def process(value):
        started.set()
        assert release.wait(3)
        if value < 0:
            raise ValueError("result failed")
        return value

    with ResultWorker(process) as worker:
        worker.submit(1)
        assert started.wait(3)
        worker.submit(2)  # Caller can advance while first result is still blocked.
        with pytest.raises(RuntimeError, match="Drain"):
            worker.submit(3)
        release.set()
        assert worker.pop() == 1
        assert worker.pop() == 2
        worker.submit(-1)
        with pytest.raises(ValueError, match="result failed"):
            worker.pop()


def test_reusable_batch_buffers_are_bounded(tmp_path, monkeypatch):
    items = inputs(tmp_path, 21)
    stats = {}
    # Keep the test about ownership/reuse, not expensive image interpolation.
    monkeypatch.setattr(streaming, "prepare_image", lambda data, tta, out, **kwargs: out[0].fill(len(data)))
    pointers = set()
    count = 0
    with closing(streaming.prepared_stream(items, 2, prefetch_batches=2, reuse_buffers=True, stats=stats)) as batches:
        for offset, views in batches:
            pointers.add(views[0].ctypes.data)
            assert offset == count
            count += len(views[0])
    assert count == 21
    assert len(pointers) <= 4
    assert stats["host_buffer_allocations"] <= 4


@pytest.mark.parametrize("compact", [False, True])
def test_cuda_device_slots_and_pinned_source_lifetime(compact):
    import torch

    from deployment.mambo_deploy.transfers import device_batches, download_tensors, pinned_factory

    if not torch.cuda.is_available():
        pytest.skip("Intentional CUDA test; set CUDA_VISIBLE_DEVICES")
    allocate = pinned_factory("cuda:0", compact=compact)
    host = allocate((2, 3, 4, 4))
    assert torch.from_numpy(host).is_pinned()

    def source():
        for i in range(7):
            host.fill(i)
            yield i * 2, (host[:1] if i == 6 else host,)

    stats = {}
    pointers = set()
    with closing(device_batches(source(), "torch", "cuda:0", stats)) as batches:
        for offset, views, count in batches:
            assert views[0].dtype == (torch.uint8 if compact else torch.float32)
            pointers.add(views[0].data_ptr())
            result = download_tensors([views[0] * 2], torch=torch)[0]
            np.testing.assert_array_equal(result, np.full((count, 3, 4, 4), offset, dtype=np.float32))
    assert len(pointers) == 2
    assert stats["device_buffer_allocations"] == 2
    assert stats["h2d_device_seconds"] >= 0


def test_cuda_deferred_download_retains_outputs_after_slot_reuse():
    import torch

    from deployment.mambo_deploy.transfers import download_tensors

    if not torch.cuda.is_available():
        pytest.skip("Intentional CUDA test; set CUDA_VISIBLE_DEVICES")
    stream = torch.cuda.Stream()
    source = torch.empty((8, 128), device="cuda")
    pending = []
    for index in range(6):
        source.fill_(index)
        # Completing later must retain this batch, not expose a reused device slot.
        pending.append(download_tensors([source[: index + 1], source.sum(1)], torch=torch, defer=True, stream=stream))
    del source
    for index, complete in enumerate(pending):
        values, sums = complete()
        np.testing.assert_array_equal(values, np.full((index + 1, 128), index))
        np.testing.assert_array_equal(sums, np.full(8, index * 128))


@pytest.mark.parametrize("tta", [None, resolve_tta("rotation30_pad25_3")])
def test_compact_stream_preserves_views_and_quarters_storage(tmp_path, tta):
    import torch

    from deployment.mambo_deploy.preprocessing import TorchPreprocess

    finish = TorchPreprocess(torch, "cpu")
    paths = inputs(tmp_path, 5)
    for offset, views in streaming.prepared_stream(paths, 2, compact=True):
        for index, view in enumerate(views):
            assert view.dtype == np.uint8
            expected = np.stack([streaming.prepare_image(p.read_bytes(), tta)[index] for p, _ in paths[offset : offset + len(view)]])
            assert view.nbytes * 4 == expected.nbytes
            np.testing.assert_allclose(finish(torch.from_numpy(view)).numpy(), expected, atol=1e-6)
