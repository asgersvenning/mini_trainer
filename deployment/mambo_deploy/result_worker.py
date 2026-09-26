"""Ordered result processing, overlapped with inference and bounded to two batches."""

from collections import deque
from concurrent.futures import ThreadPoolExecutor


class ResultWorker:
    def __init__(self, function):
        self.function = function
        self.pending = deque()
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mambo-results")

    def __enter__(self):
        return self

    def submit(self, *args):
        if len(self.pending) >= 2:
            raise RuntimeError("Drain a result before submitting another batch")
        self.pending.append(self.pool.submit(self.function, *args))

    def pop(self):
        return self.pending.popleft().result()

    def __exit__(self, *exc):
        for future in self.pending:
            future.cancel()
        self.pool.shutdown(wait=True, cancel_futures=True)
