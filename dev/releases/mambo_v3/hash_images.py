"""Concurrent image reads for cache warmup, with resumable verified hashes."""

import hashlib
import json
import os
import sqlite3
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from dev.benchmarks.inference.onnx_inference import file_hash


def default_workers():
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
    return min(256, max(8, 2 * cpus))


def hash_images(records, root, checkpoint, provenance, workers=None):
    """Preserve record ordering; overlap cold reads without queuing the whole dataset."""
    workers = default_workers() if workers is None else workers
    if workers < 1:
        raise ValueError("Hash workers must be positive")
    root = Path(root).resolve()
    identity = hashlib.sha256(json.dumps([str(root), provenance, records], sort_keys=True).encode()).hexdigest()
    connection = sqlite3.connect(checkpoint)
    try:
        connection.execute("CREATE TABLE IF NOT EXISTS identity (value TEXT NOT NULL)")
        previous = connection.execute("SELECT value FROM identity").fetchone()
        if previous and previous[0] != identity:
            raise ValueError("Hash checkpoint belongs to different inputs; use a fresh cache")
        if not previous:
            connection.execute("INSERT INTO identity VALUES (?)", (identity,))
        connection.execute("CREATE TABLE IF NOT EXISTS hashes (idx INTEGER PRIMARY KEY, digest TEXT, size INTEGER, mtime INTEGER)")
        connection.commit()
        cached = {i: (digest, size, mtime) for i, digest, size, mtime in connection.execute("SELECT * FROM hashes")}

        def read(index):
            path = (root / records[index]["path"]).resolve()
            if not path.is_relative_to(root):
                raise ValueError(f"Unsafe image path: {records[index]['path']}")
            before = path.stat()
            stamp = (before.st_size, before.st_mtime_ns)
            prior = cached.get(index)
            if prior and prior[1:] == stamp:
                return index, prior[0], *stamp, True
            digest = file_hash(path)
            after = path.stat()
            if (after.st_size, after.st_mtime_ns) != stamp:
                raise ValueError(f"Image changed while hashing: {path}")
            return index, digest, *stamp, False

        done_count = reused = 0
        started = reported = time.monotonic()
        print(f"Hashing {len(records):,} images with {workers} concurrent readers; checkpoint: {checkpoint}", flush=True)
        indices = iter(range(len(records)))
        pending = set()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            try:
                for index in indices:
                    pending.add(pool.submit(read, index))
                    if len(pending) >= workers * 2:
                        break
                while pending:
                    finished, pending = wait(pending, timeout=5, return_when=FIRST_COMPLETED)
                    for future in finished:
                        index, digest, size, mtime, was_cached = future.result()
                        records[index]["sha256"] = digest
                        if not was_cached:
                            connection.execute("INSERT OR REPLACE INTO hashes VALUES (?, ?, ?, ?)", (index, digest, size, mtime))
                        done_count += 1
                        reused += was_cached
                        following = next(indices, None)
                        if following is not None:
                            pending.add(pool.submit(read, following))
                    now = time.monotonic()
                    if now - reported >= 5 or not pending:
                        connection.commit()
                        elapsed = max(now - started, 0.001)
                        print(
                            f"Hashed {done_count:,}/{len(records):,} ({reused:,} reused); "
                            f"{done_count / elapsed:.1f} images/s; {len(pending)} queued/running; {elapsed:.0f}s elapsed",
                            flush=True,
                        )
                        reported = now
            finally:
                connection.commit()
                for future in pending:
                    future.cancel()
    finally:
        connection.close()
