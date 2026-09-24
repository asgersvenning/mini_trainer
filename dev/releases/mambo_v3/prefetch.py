"""Bounded, ordered CPU preparation for release collection; no runtime calls in workers."""

import hashlib
import io
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from functools import partial

import numpy as np
from PIL import Image

from deployment.mambo_deploy.augmentation import _prepare_view
from deployment.mambo_deploy.preprocessing import _rgb, preprocess


def prepare_record(record, root, tta):
    path = root / record["path"]
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != record["sha256"]:
        raise ValueError(f"Image bytes changed: {path}")
    with Image.open(io.BytesIO(data)) as image:
        decoded = _rgb(image)
    if tta is None:
        return (preprocess(decoded),)
    return tuple(_prepare_view(decoded, transform) for transform in tta.transforms)


def prepared_batches(records, root, batch_size, workers, prefetch_batches, tta=None):
    """Keep at most prefetch_batches queued batches plus the batch held by the consumer."""
    if batch_size < 1 or workers < 0 or prefetch_batches < 0:
        raise ValueError("Positive batch size and nonnegative workers/prefetch required")
    with ExitStack() as stack:
        pool = stack.enter_context(ThreadPoolExecutor(max_workers=workers)) if workers else None
        prepare = partial(prepare_record, root=root, tta=tta)

        def batch(offset):
            start = time.perf_counter()
            selected = records[offset : offset + batch_size]
            images = list(pool.map(prepare, selected)) if pool else [prepare(record) for record in selected]
            views = tuple(np.stack(view) for view in zip(*images, strict=True))
            return offset, selected, views, time.perf_counter() - start

        offsets = iter(range(0, len(records), batch_size))
        if not prefetch_batches:
            for offset in offsets:
                yield batch(offset)
            return
        # A single coordinator uses the loading pool; no GPU/model state crosses threads.
        producer = ThreadPoolExecutor(max_workers=1)
        pending = deque()
        try:
            for _ in range(prefetch_batches):
                if (offset := next(offsets, None)) is not None:
                    pending.append(producer.submit(batch, offset))
            while pending:
                result = pending.popleft().result()
                if (offset := next(offsets, None)) is not None:
                    pending.append(producer.submit(batch, offset))
                yield result
                del result
        finally:
            for future in pending:
                future.cancel()
            producer.shutdown(wait=True, cancel_futures=True)
