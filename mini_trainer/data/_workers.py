import os


def _available_cpu_count() -> int:
    """Use process limits when available, falling back to the host CPU count."""
    counts = []
    try:
        count = os.process_cpu_count()  # Python >=3.13
        if count is not None:
            counts.append(count)
    except (AttributeError, OSError, NotImplementedError):
        pass
    try:
        counts.append(len(os.sched_getaffinity(0)))
    except (AttributeError, OSError, NotImplementedError):
        pass
    return min(counts) if counts else (os.cpu_count() or 0)


def _default_worker_count(cap: int, reserve: int = 4, minimum: int = 0) -> int:
    count = max(0, _available_cpu_count() - reserve)
    return max(minimum, min(cap, count - count % 2))
