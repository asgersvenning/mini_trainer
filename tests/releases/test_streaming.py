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


def test_ready_work_prioritizes_earliest_batch(tmp_path, monkeypatch):
    import time
    from concurrent.futures import ThreadPoolExecutor

    items = inputs(tmp_path, 4)
    indices = {p.read_bytes(): i for i, (p, _) in enumerate(items)}
    preparing_first, release_first, later_read = threading.Event(), threading.Event(), threading.Event()
    original = streaming.read_image
    order, stats = [], {}

    def read(path, size, digest):
        index = int(path.stem)
        if index:
            assert preparing_first.wait(3)
        if index == 1:
            assert later_read.wait(3)
        data = original(path, size, digest)
        if index == 3:
            later_read.set()
        return data

    def prepare(data, tta, out, **kwargs):
        index = indices[data]
        order.append(index)
        if index == 0:
            preparing_first.set()
            assert release_first.wait(5)
        out[0].fill(index)

    monkeypatch.setattr(streaming, "read_image", read)
    monkeypatch.setattr(streaming, "prepare_image", prepare)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(list, streaming.prepared_stream(items, 2, read_workers=4, prepare_workers=1, read_window=4, stats=stats))
        try:
            deadline = time.monotonic() + 3
            while stats.get("encoded_ready", 0) != 3 and time.monotonic() < deadline:
                time.sleep(0.001)
            assert stats.get("encoded_ready") == 3
        finally:
            release_first.set()
        assert len(future.result(timeout=5)) == 2
    assert order == [0, 1, 2, 3]


def test_lease_prevents_early_reuse_and_source_errors_propagate(tmp_path, monkeypatch):
    items = inputs(tmp_path, 12)
    second_prepared = threading.Event()
    original = streaming.prepare_image
    second_data = items[3][0].read_bytes()

    def prepare(data, tta, out=None, **kwargs):
        result = original(data, tta, out=out, **kwargs)
        if data == second_data:
            second_prepared.set()
        return result

    monkeypatch.setattr(streaming, "prepare_image", prepare)
    with closing(streaming.prepared_stream(items, 2, prefetch_batches=1, reuse_buffers=True)) as stream:
        _, first = next(stream)
        assert second_prepared.wait(3)
        # Preparing the next batch must not overwrite the batch still held by the consumer.
        np.testing.assert_array_equal(first[0], np.stack([preprocess(p) for p, _ in items[:2]]))
        _, second = next(stream)
        np.testing.assert_array_equal(second[0][0], preprocess(items[2][0]))

    def broken():
        yield items[0]
        raise OSError("input iterator failed")

    with pytest.raises(OSError, match="input iterator failed"):
        list(streaming.prepared_stream(broken(), 2))
    assert not any(t.name.startswith(("mambo-read", "mambo-prepare", "mambo-stream")) for t in threading.enumerate())


def test_metadata_latency_does_not_block_preparation(tmp_path, monkeypatch):
    from pathlib import Path

    items = inputs(tmp_path, 5)
    prepared = threading.Event()
    original_stat, original_prepare = Path.stat, streaming.prepare_image
    threads = []

    def stat(path, *args, **kwargs):
        if path in [p for p, _ in items]:
            threads.append(threading.current_thread().name)
            if path == items[2][0]:
                assert prepared.wait(3), "metadata IO must not stall the preparation owner"
        return original_stat(path, *args, **kwargs)

    def prepare(data, tta, out=None, **kwargs):
        result = original_prepare(data, tta, out=out, **kwargs)
        prepared.set()
        return result

    monkeypatch.setattr(Path, "stat", stat)
    monkeypatch.setattr(streaming, "prepare_image", prepare)
    assert len(list(streaming.prepared_stream(items, 2, read_workers=4, read_window=8))) == 3
    assert all(name.startswith("mambo-read") for name in threads)


def test_close_with_pending_byte_admission(tmp_path):
    items = inputs(tmp_path, 8)
    budget = max(p.stat().st_size for p, _ in items)
    with closing(streaming.prepared_stream(items, 1, read_workers=4, read_window=8, prefetch_batches=0, encoded_budget=budget)) as stream:
        next(stream)
    assert not any(t.name.startswith(("mambo-read", "mambo-prepare", "mambo-stream")) for t in threading.enumerate())


def test_invalid_source_fails_before_starting_workers():
    class InvalidSource:
        def __iter__(self):
            raise ValueError("cannot iterate source")

    with pytest.raises(ValueError, match="cannot iterate source"):
        next(streaming.prepared_stream(InvalidSource(), 1))


def test_full_byte_budget_does_not_occupy_io_workers(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    items = inputs(tmp_path, 6)
    budget = max(p.stat().st_size for p, _ in items)
    inspected, reads = [], []
    all_inspected, release_prepare = threading.Event(), threading.Event()
    original_metadata, original_read = streaming.image_metadata, streaming.read_image

    def metadata(path, digest):
        result = original_metadata(path, digest)
        inspected.append(path)
        if len(inspected) == len(items):
            all_inspected.set()
        return result

    def read(path, size, digest):
        reads.append(path)
        return original_read(path, size, digest)

    def prepare(data, tta, out, **kwargs):
        assert release_prepare.wait(5)
        out[0].fill(0)

    monkeypatch.setattr(streaming, "image_metadata", metadata)
    monkeypatch.setattr(streaming, "read_image", read)
    monkeypatch.setattr(streaming, "prepare_image", prepare)
    stats = {}
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            list,
            streaming.prepared_stream(items, 2, read_workers=2, read_window=6, encoded_budget=budget, stats=stats),
        )
        try:
            assert all_inspected.wait(3), "byte-budget backpressure must leave IO workers available for metadata"
            assert len(reads) <= 1  # The reservation includes bytes held by preparation.
        finally:
            release_prepare.set()
        assert len(future.result(timeout=5)) == 3
    assert stats["peak_encoded_bytes"] <= budget
    assert stats["encoded_bytes"] == 0
