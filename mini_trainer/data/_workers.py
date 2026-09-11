import os
import re
from pathlib import Path


def _read_text(path):
    try:
        return path.read_text()
    except (OSError, UnicodeError):
        return ""


def _cgroup_cpu_count():
    """Smallest visible v1/v2 bandwidth quota, including ancestor limits."""
    groups = {}
    for line in _read_text(Path("/proc/self/cgroup")).splitlines():
        fields = line.split(":", 2)
        if len(fields) == 3:
            for controller in fields[1].split(","):
                groups[controller] = fields[2]
    limits = []
    for line in _read_text(Path("/proc/self/mountinfo")).splitlines():
        left, separator, right = line.partition(" - ")
        fields, filesystem = left.split(), right.split()
        if not separator or len(fields) < 5 or len(filesystem) < 3:
            continue
        if filesystem[0] == "cgroup2":
            group = groups.get("")
            filenames = ("cpu.max",)
        elif filesystem[0] == "cgroup" and "cpu" in filesystem[2].split(","):
            group = groups.get("cpu")
            filenames = ("cpu.cfs_quota_us", "cpu.cfs_period_us")
        else:
            continue
        if group is None:
            continue
        root, mount = (Path(re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), value)) for value in fields[3:5])
        group = Path(group)
        if not all(path.is_absolute() and ".." not in path.parts for path in (root, mount, group)):
            continue
        try:
            relative = group.relative_to(root)
        except ValueError:
            # A cgroup namespace can expose paths relative to its own root.
            relative = group.relative_to("/")
        current = mount / relative
        while True:
            values = " ".join(_read_text(current / name) for name in filenames).split()
            if len(values) == 2:
                try:
                    quota, period = map(int, values)
                    if quota > 0 and period > 0:
                        limits.append(quota // period)
                except ValueError:
                    pass  # v2 "max", or malformed/unavailable quota data.
            if current == mount:
                break
            current = current.parent
    return min(limits) if limits else None


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
    if not counts:
        counts.append(os.cpu_count() or 0)
    quota = _cgroup_cpu_count()
    if quota is not None:
        counts.append(quota)
    try:
        allocated = int(os.environ.get("SLURM_CPUS_PER_TASK", ""))
        if allocated > 0:
            counts.append(allocated)
    except ValueError:
        pass
    return min(counts)


def _default_worker_count(cap: int, reserve: int = 4, minimum: int = 0) -> int:
    count = max(0, _available_cpu_count() - reserve)
    return max(minimum, min(cap, count - count % 2))
