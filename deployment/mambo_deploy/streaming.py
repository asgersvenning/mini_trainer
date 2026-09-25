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
from .preprocessing import _rgb, preprocess


def read_image(path, size, digest):
    with path.open("rb") as stream:
        data = stream.read(size + 1)
    if len(data) != size:
        raise ValueError(f"Image size changed: {path}")
    if digest is not None and hashlib.sha256(data).hexdigest() != digest:
        raise ValueError(f"Image bytes changed: {path}")
    return data


def prepare_image(data, tta):
    with Image.open(io.BytesIO(data)) as image:
        decoded = _rgb(image)
    return (preprocess(decoded),) if tta is None else tuple(_prepare_view(decoded, view) for view in tta.transforms)


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


def assemble_batch(images, pool=None):
    """Stack and release per-image arrays on the assembler thread."""
    start = time.perf_counter()
    if pool is None:
        views = tuple(np.stack(view) for view in zip(*images, strict=True))
    else:
        shapes = [(len(images), *view.shape) for view in images[0]]
        buffers = pool.acquire(shapes)
        views = tuple(buffer[: len(images)] for buffer in buffers)
        for index, view in enumerate(zip(*images, strict=True)):
            np.stack(view, out=views[index])
    images.clear()
    return views, time.perf_counter() - start


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
    """Yield (offset, stacked views) from (path, optional SHA256) items; close on early exit.

    IO reservations include in-flight reads and bytes held by preparation. Prepared
    images are bounded by (prefetch_batches + 1) * batch_size, plus a stacked batch.
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

    def produce():
        reads, decoding, buffers, sizes = {}, {}, {}, {}
        ready = {}
        next_index, reserved, assemble_offset = 0, 0, 0
        assembling = None
        assembling_count = 0
        pending = None
        exhausted = False
        readers = ThreadPoolExecutor(max_workers=read_workers)
        preparers = ThreadPoolExecutor(max_workers=prepare_workers)
        assembler = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mambo-assemble")
        try:
            while True:
                with condition:
                    if state["stop"]:
                        break
                    consumed = state["consumed"]
                for index, future in list(reads.items()):
                    if future.done():
                        buffers[index] = future.result()
                        del reads[index]
                for index, future in list(decoding.items()):
                    if future.done():
                        result = future.result()
                        del decoding[index]
                        reserved -= sizes.pop(index)
                        ready[index] = result
                        del result
                for index in sorted(buffers):
                    if index < consumed + capacity and len(decoding) < prepare_workers:
                        decoding[index] = preparers.submit(prepare_image, buffers.pop(index), tta)
                while not exhausted and next_index < consumed + read_window and len(reads) < read_workers:
                    if pending is None:
                        try:
                            path, digest = next(source)
                        except StopIteration:
                            exhausted = True
                            with condition:
                                state["end"] = next_index
                                condition.notify_all()
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
                    reads[next_index] = readers.submit(read_image, path, size, digest)
                    next_index += 1
                    pending = None
                if assembling is not None and assembling.done():
                    views, elapsed = assembling.result()
                    with condition:
                        batches[assemble_offset] = views
                        stats["batch_assembly_seconds"] = stats.get("batch_assembly_seconds", 0.0) + elapsed
                        condition.notify_all()
                    del views
                    assembling = None
                    assemble_offset += assembling_count
                    assembling_count = 0
                end = min(assemble_offset + batch_size, next_index) if exhausted else assemble_offset + batch_size
                if assembling is None and end > assemble_offset and all(i in ready for i in range(assemble_offset, end)):
                    assembling_count = end - assemble_offset
                    images = [ready.pop(i) for i in range(assemble_offset, end)]
                    assembling = (
                        assembler.submit(assemble_batch, images, buffer_pool) if buffer_pool else assembler.submit(assemble_batch, images)
                    )
                with condition:
                    prepared_count = len(ready) + sum(len(views[0]) for views in batches.values()) + assembling_count
                    stats.update(
                        reading=len(reads),
                        encoded_ready=len(buffers),
                        preparing=len(decoding),
                        prepared_images=prepared_count,
                        prepared_batches=len(batches),
                        assembling=assembling_count,
                        encoded_bytes=reserved,
                        host_buffer_allocations=buffer_pool.allocations if buffer_pool else None,
                    )
                    stats["peak_encoded_bytes"] = max(stats.get("peak_encoded_bytes", 0), reserved)
                    stats["peak_prepared_images"] = max(stats.get("peak_prepared_images", 0), prepared_count + len(decoding))
                    if exhausted and not reads and not decoding and not buffers and not ready and assembling is None:
                        break
                    condition.wait(timeout=0.005)
        except BaseException as error:
            with condition:
                state["error"] = error
                condition.notify_all()
        finally:
            for future in (*reads.values(), *decoding.values()):
                future.cancel()
            readers.shutdown(wait=True, cancel_futures=True)
            preparers.shutdown(wait=True, cancel_futures=True)
            assembler.shutdown(wait=True, cancel_futures=True)

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
                condition.notify_all()
    finally:
        with condition:
            state["stop"] = True
            condition.notify_all()
        producer.join()
