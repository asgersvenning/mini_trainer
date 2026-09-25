"""Bounded path streaming with independent IO and image preparation concurrency."""

import hashlib
import io
import threading
import time
from concurrent.futures import CancelledError, ThreadPoolExecutor
from heapq import heappop, heappush
from pathlib import Path
from queue import Empty, SimpleQueue

import numpy as np
from PIL import Image

from .augmentation import _prepare_view
from .preprocessing import RECIPE, _rgb, prepare_uint8, preprocess


def read_image(path, size, digest):
    with path.open("rb") as stream:
        data = stream.read(size + 1)
    if len(data) != size:
        raise ValueError(f"Image size changed: {path}")
    if digest is not None and hashlib.sha256(data).hexdigest() != digest:
        raise ValueError(f"Image bytes changed: {path}")
    return data


def prepare_image(data, tta, out=None, *, compact=False):
    with Image.open(io.BytesIO(data)) as image:
        decoded = _rgb(image)
    prepare = prepare_uint8 if compact else preprocess
    if out is None:
        return (prepare(decoded),) if tta is None else tuple(_prepare_view(decoded, view, compact=compact) for view in tta.transforms)
    if tta is None:
        prepare(decoded, out=out[0])
    else:
        for transform, target in zip(tta.transforms, out, strict=True):
            _prepare_view(decoded, transform, out=target, compact=compact)
    return out


class EncodedReads:
    """Own byte reservations in input order; metadata and file IO stay in readers."""

    def __init__(self, budget, stats):
        self.budget, self.stats = budget, stats
        self.condition = threading.Condition()
        self.next_index = self.reserved = 0
        self.closed = False
        stats["encoded_bytes"] = 0

    def read(self, index, path, digest):
        path = Path(path)
        size = path.stat().st_size
        if size > self.budget:
            raise ValueError(f"Image exceeds encoded byte budget: {path}")
        with self.condition:
            self.condition.wait_for(lambda: self.closed or (index == self.next_index and self.reserved + size <= self.budget))
            if self.closed:
                raise CancelledError()
            self.reserved += size
            self.stats["encoded_bytes"] = self.reserved
            self.stats["peak_encoded_bytes"] = max(self.stats.get("peak_encoded_bytes", 0), self.reserved)
            self.next_index += 1
            self.condition.notify_all()
        return size, read_image(path, size, digest)

    def release(self, size):
        with self.condition:
            self.reserved -= size
            self.stats["encoded_bytes"] = self.reserved
            self.condition.notify_all()

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()


def prepared_stream(
    items,
    batch_size,
    *,
    tta=None,
    read_workers=32,
    prepare_workers=4,
    read_window=128,
    prefetch_batches=2,
    encoded_budget=256 * 1024**2,
    stats=None,
    reuse_buffers=False,
    buffer_factory=None,
    compact=False,
):
    """Yield ordered (offset, batch views); reusable buffers are leased until next().

    The reading stage owns byte admission. One preparation owner assigns disjoint
    slices and recycles returned batches; workers report completions. Encoded bytes
    stay reserved through preparation; batch capacity is released by the consumer.
    Close on early exit. Shutdown waits for running filesystem/preparation calls.
    """
    values = (batch_size, read_workers, prepare_workers, read_window, encoded_budget)
    if any(not isinstance(v, int) or v < 1 for v in values) or not isinstance(prefetch_batches, int) or prefetch_batches < 0:
        raise ValueError("Positive batch/workers/window/budget and nonnegative prefetch required")
    if read_window < batch_size:
        raise ValueError("read_window must cover at least one batch")
    stats = {} if stats is None else stats
    events, output = SimpleQueue(), SimpleQueue()
    source = iter(items)
    capacity = batch_size * (prefetch_batches + 1)
    factory = buffer_factory or (lambda shape: np.empty(shape, dtype=np.uint8 if compact else np.float32))
    shape = (batch_size, 3, RECIPE["crop_size"], RECIPE["crop_size"])

    def prepare_into(data, target):
        start = time.perf_counter()
        prepare_image(data, tta, out=target, compact=compact)
        return time.perf_counter() - start

    def submit(pool, kind, index, size, function, *args):
        pool.submit(function, *args).add_done_callback(lambda future: events.put((kind, index, size, future)))

    def produce():
        ready, available, active = [], [], {}
        consumed = admitted = emitted = 0
        reading = preparing = 0
        exhausted = False
        readers = ThreadPoolExecutor(max_workers=read_workers, thread_name_prefix="mambo-read")
        preparers = ThreadPoolExecutor(max_workers=prepare_workers, thread_name_prefix="mambo-prepare")
        encoded = EncodedReads(encoded_budget, stats)
        stats.update(prepared_images=0, prepared_batches=0, host_buffer_allocations=0)

        def complete(event):
            nonlocal consumed, reading, preparing
            kind, index, size, value = event
            if kind == "read":
                reading -= 1
                size, data = value.result()
                heappush(ready, (index, size, data))
            elif kind == "prepared":
                stats["preparation_worker_seconds"] = stats.get("preparation_worker_seconds", 0.0) + value.result()
                preparing -= 1
                encoded.release(size)
                active[index // batch_size][1] += 1
                stats["prepared_images"] += 1
            elif kind == "taken":
                stats["prepared_images"] -= size
                stats["prepared_batches"] -= 1
            elif kind == "returned":
                consumed = index + size
                if reuse_buffers:
                    available.append(value)
            return kind != "stop"

        try:
            while True:
                # Completions identify exactly which item changed; no future/buffer scans.
                try:
                    while complete(events.get_nowait()):
                        pass
                    break
                except Empty:
                    pass
                while ready and ready[0][0] < consumed + capacity and preparing < prepare_workers:
                    index, size, data = heappop(ready)
                    number, slot = divmod(index, batch_size)
                    if number not in active:
                        if available:
                            views = available.pop()
                        else:
                            views = tuple(factory(shape) for _ in range(1 if tta is None else len(tta.transforms)))
                            stats["host_buffer_allocations"] += len(views)
                        active[number] = [views, 0]
                    target = tuple(view[slot] for view in active[number][0])
                    submit(preparers, "prepared", index, size, prepare_into, data, target)
                    preparing += 1
                    del data
                while not exhausted and admitted < consumed + read_window and reading < read_workers:
                    try:
                        path, digest = next(source)
                    except StopIteration:
                        exhausted = True
                        break
                    submit(readers, "read", admitted, 0, encoded.read, admitted, path, digest)
                    reading += 1
                    admitted += 1
                while emitted // batch_size in active:
                    views, count = active[emitted // batch_size]
                    expected = min(batch_size, admitted - emitted) if exhausted else batch_size
                    if count != expected:
                        break
                    del active[emitted // batch_size]
                    stats["prepared_batches"] += 1
                    output.put((emitted, tuple(view[:count] for view in views)))
                    emitted += count
                stats.update(reading=reading, encoded_ready=len(ready), preparing=preparing)
                stats["peak_prepared_images"] = max(stats.get("peak_prepared_images", 0), stats["prepared_images"] + preparing)
                if exhausted and consumed == admitted:
                    output.put(None)
                    return
                if not complete(events.get()):
                    break
        except BaseException as error:
            output.put(error)
        finally:
            encoded.close()
            readers.shutdown(wait=True, cancel_futures=True)
            preparers.shutdown(wait=True, cancel_futures=True)
            # Completed futures can retain encoded images; discard them after shutdown.
            while not events.empty():
                events.get_nowait()

    producer = threading.Thread(target=produce, name="mambo-stream")
    producer.start()
    try:
        while True:
            start = time.perf_counter()
            result = output.get()
            stats["queue_wait_seconds"] = stats.get("queue_wait_seconds", 0.0) + time.perf_counter() - start
            stats["input_wait_seconds"] = stats["queue_wait_seconds"]
            if isinstance(result, BaseException):
                raise result
            if result is None:
                return
            offset, views = result
            events.put(("taken", offset, len(views[0]), None))
            yield offset, views
            events.put(("returned", offset, len(views[0]), views))
    finally:
        events.put(("stop", 0, 0, None))
        producer.join()
