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
):
    """Yield (offset, stacked views) from (path, optional SHA256) items; close on early exit.

    IO reservations include in-flight reads and bytes held by preparation. Prepared
    images are bounded by (prefetch_batches + 1) * batch_size, plus a stacked batch.
    Workers never touch the runtime. Shutdown waits for already-running filesystem calls.
    """
    values = (batch_size, read_workers, prepare_workers, read_window, encoded_budget)
    if any(not isinstance(v, int) or v < 1 for v in values) or not isinstance(prefetch_batches, int) or prefetch_batches < 0:
        raise ValueError("Positive batch/workers/window/budget and nonnegative prefetch required")
    if read_window < batch_size:
        raise ValueError("read_window must cover at least one batch")
    stats = {} if stats is None else stats
    condition = threading.Condition()
    state = dict(stop=False, consumed=0, end=None, error=None)
    ready = {}
    source = iter(items)
    capacity = batch_size * (prefetch_batches + 1)

    def produce():
        reads, decoding, buffers, sizes = {}, {}, {}, {}
        next_index, reserved = 0, 0
        pending = None
        exhausted = False
        readers = ThreadPoolExecutor(max_workers=read_workers)
        preparers = ThreadPoolExecutor(max_workers=prepare_workers)
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
                        with condition:
                            ready[index] = result
                            condition.notify_all()
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
                with condition:
                    stats.update(
                        reading=len(reads),
                        encoded_ready=len(buffers),
                        preparing=len(decoding),
                        prepared_images=len(ready),
                        encoded_bytes=reserved,
                    )
                    stats["peak_encoded_bytes"] = max(stats.get("peak_encoded_bytes", 0), reserved)
                    stats["peak_prepared_images"] = max(stats.get("peak_prepared_images", 0), len(ready) + len(decoding))
                    if exhausted and not reads and not decoding and not buffers:
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
                    if all(index in ready for index in range(offset, end)):
                        images = [ready.pop(index) for index in range(offset, end)]
                        break
                    condition.wait()
            views = tuple(np.stack(view) for view in zip(*images, strict=True))
            del images
            with condition:
                stats["prepared_images"] = len(ready)
                stats["input_wait_seconds"] = stats.get("input_wait_seconds", 0.0) + time.perf_counter() - start
                condition.notify_all()
            yield offset, views
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
