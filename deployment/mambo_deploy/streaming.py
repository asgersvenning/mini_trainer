"""Bounded path streaming with independent IO and image preparation concurrency."""

import hashlib
import io
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from .augmentation import _prepare_view
from .preprocessing import RECIPE, _rgb, preprocess


def read_image(path, size, digest):
    with path.open("rb") as stream:
        data = stream.read(size + 1)
    if len(data) != size:
        raise ValueError(f"Image size changed: {path}")
    if digest is not None and hashlib.sha256(data).hexdigest() != digest:
        raise ValueError(f"Image bytes changed: {path}")
    return data


def prepare_image(data, tta, out=None):
    with Image.open(io.BytesIO(data)) as image:
        decoded = _rgb(image)
    if out is None:
        return (preprocess(decoded),) if tta is None else tuple(_prepare_view(decoded, view) for view in tta.transforms)
    if tta is None:
        preprocess(decoded, out=out[0])
    else:
        for transform, target in zip(tta.transforms, out, strict=True):
            _prepare_view(decoded, transform, out=target)
    return out


class BatchBuffers:
    def __init__(self, factory=None):
        self.factory = factory or (lambda shape: np.empty(shape, dtype=np.float32))
        self.available = []
        self.lock = threading.Lock()
        self.allocations = 0

    def acquire(self, shapes):
        with self.lock:
            for i, buffers in enumerate(self.available):
                if all(b.shape[0] >= s[0] and b.shape[1:] == s[1:] for b, s in zip(buffers, shapes, strict=True)):
                    return self.available.pop(i)
            self.allocations += len(shapes)
        return tuple(self.factory(shape) for shape in shapes)

    def release(self, views):
        # Partial batches retain the full allocation via .base; no further work follows the final partial batch.
        with self.lock:
            self.available.append(views)


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
):
    """Yield (offset, batch views) from (path, optional SHA256) items; close on early exit.

    IO reservations include in-flight reads and bytes held by preparation. Prepared
    images are bounded by (prefetch_batches + 1) * batch_size, plus the consumed batch.
    Workers write directly to disjoint batch slices; completion callbacks wake the coordinator.
    Preparation workers do not call models or sessions. Shutdown waits for running filesystem calls.
    """
    values = (batch_size, read_workers, prepare_workers, read_window, encoded_budget)
    if any(not isinstance(v, int) or v < 1 for v in values) or not isinstance(prefetch_batches, int) or prefetch_batches < 0:
        raise ValueError("Positive batch/workers/window/budget and nonnegative prefetch required")
    if read_window < batch_size:
        raise ValueError("read_window must cover at least one batch")
    stats = {} if stats is None else stats
    condition = threading.Condition()
    state = dict(stop=False, consumed=0, end=None, error=None)
    batches = {}
    buffer_pool = BatchBuffers(buffer_factory) if reuse_buffers else None
    source = iter(items)
    capacity = batch_size * (prefetch_batches + 1)

    def wake(_=None):
        with condition:
            state["generation"] += 1
            condition.notify_all()

    state["generation"] = 0

    def produce():
        reads, decoding, buffers, sizes, active = {}, {}, {}, {}, {}
        next_index, reserved = 0, 0
        pending = None
        exhausted = False
        readers = ThreadPoolExecutor(max_workers=read_workers, thread_name_prefix="mambo-read")
        preparers = ThreadPoolExecutor(max_workers=prepare_workers, thread_name_prefix="mambo-prepare")
        shape = (batch_size, 3, RECIPE["crop_size"], RECIPE["crop_size"])
        shapes = [shape] * (1 if tta is None else len(tta.transforms))
        pool = buffer_pool or BatchBuffers(buffer_factory)
        try:
            while True:
                with condition:
                    if state["stop"]:
                        break
                    generation, consumed = state["generation"], state["consumed"]
                for index, future in list(reads.items()):
                    if future.done():
                        buffers[index] = future.result()
                        del reads[index]
                for index, future in list(decoding.items()):
                    if future.done():
                        elapsed = future.result()
                        del decoding[index]
                        reserved -= sizes.pop(index)
                        active[index // batch_size][1] += 1
                        stats["preparation_worker_seconds"] = stats.get("preparation_worker_seconds", 0.0) + elapsed
                for number, (views, count) in list(active.items()):
                    expected = min(batch_size, next_index - number * batch_size) if exhausted else batch_size
                    if count == expected:
                        with condition:
                            batches[number * batch_size] = tuple(view[:count] for view in views)
                            condition.notify_all()
                        del active[number]
                for index in list(buffers):
                    if index < consumed + capacity and len(decoding) < prepare_workers:
                        number, slot = divmod(index, batch_size)
                        if number not in active:
                            active[number] = [pool.acquire(shapes), 0]
                        target = tuple(view[slot] for view in active[number][0])
                        future = preparers.submit(prepare_into, buffers.pop(index), target)
                        decoding[index] = future
                        future.add_done_callback(wake)
                while not exhausted and next_index < consumed + read_window and len(reads) < read_workers:
                    if pending is None:
                        try:
                            path, digest = next(source)
                        except StopIteration:
                            exhausted = True
                            with condition:
                                state["end"] = next_index
                                wake()
                            break
                        path = Path(path)
                        size = path.stat().st_size
                        if size > encoded_budget:
                            raise ValueError(f"Image exceeds encoded byte budget: {path}")
                        pending = path, digest, size
                    path, digest, size = pending
                    if reserved + size > encoded_budget:
                        break
                    sizes[next_index] = size
                    reserved += size
                    future = readers.submit(read_image, path, size, digest)
                    reads[next_index] = future
                    future.add_done_callback(wake)
                    next_index += 1
                    pending = None
                with condition:
                    prepared_count = sum(count for _, count in active.values()) + sum(len(v[0]) for v in batches.values())
                    stats.update(
                        reading=len(reads),
                        encoded_ready=len(buffers),
                        preparing=len(decoding),
                        prepared_images=prepared_count,
                        prepared_batches=len(batches),
                        encoded_bytes=reserved,
                        host_buffer_allocations=pool.allocations,
                    )
                    stats["peak_encoded_bytes"] = max(stats.get("peak_encoded_bytes", 0), reserved)
                    stats["peak_prepared_images"] = max(stats.get("peak_prepared_images", 0), prepared_count + len(decoding))
                    if exhausted and not reads and not decoding and not buffers and not active:
                        break
                    condition.wait_for(lambda: state["stop"] or state["generation"] != generation)
        except BaseException as error:
            with condition:
                state["error"] = error
                condition.notify_all()
        finally:
            for future in (*reads.values(), *decoding.values()):
                future.cancel()
            readers.shutdown(wait=True, cancel_futures=True)
            preparers.shutdown(wait=True, cancel_futures=True)

    def prepare_into(data, target):
        start = time.perf_counter()
        prepare_image(data, tta, out=target)
        return time.perf_counter() - start

    producer = threading.Thread(target=produce, name="mambo-stream")
    producer.start()
    offset = 0
    try:
        while True:
            start = time.perf_counter()
            with condition:
                while True:
                    if state["error"] is not None:
                        raise state["error"]
                    end = min(offset + batch_size, state["end"]) if state["end"] is not None else offset + batch_size
                    if end == offset:
                        return
                    if offset in batches:
                        views = batches.pop(offset)
                        stats["prepared_batches"] = len(batches)
                        stats["prepared_images"] = max(0, stats.get("prepared_images", 0) - len(views[0]))
                        break
                    condition.wait()
            with condition:
                stats["queue_wait_seconds"] = stats.get("queue_wait_seconds", 0.0) + time.perf_counter() - start
                stats["input_wait_seconds"] = stats["queue_wait_seconds"]
                condition.notify_all()
            yield offset, views
            if buffer_pool is not None:
                buffer_pool.release(views)
            del views
            offset = end
            with condition:
                state["consumed"] = end
                wake()
    finally:
        with condition:
            state["stop"] = True
            condition.notify_all()
        producer.join()
