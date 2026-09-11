"""Automatic worker budgets under container and scheduler CPU limits."""

import pytest

from mini_trainer.data import _workers


def mock_cgroup(monkeypatch, files, membership="0::/team/task", mount="/", location="/sys/fs/cgroup", version=2):
    filesystem = "cgroup2 cgroup rw" if version == 2 else "cgroup cgroup rw,cpu,cpuacct"
    escaped = location.replace(" ", r"\040")
    contents = {
        "/proc/self/cgroup": membership,
        "/proc/self/mountinfo": f"29 23 0:26 {mount} {escaped} rw - {filesystem}",
        **files,
    }
    monkeypatch.setattr(_workers, "_read_text", lambda path: contents.get(str(path), ""))


def test_cgroup_v2_respects_parent_quota(monkeypatch):
    mock_cgroup(
        monkeypatch,
        {
            "/sys/fs/cgroup/team/task/cpu.max": "max 100000",
            "/sys/fs/cgroup/team/cpu.max": "250000 100000",
            "/sys/fs/cgroup/cpu.max": "1600000 100000",
        },
    )
    assert _workers._cgroup_cpu_count() == 2


@pytest.mark.parametrize(
    "value,expected",
    [("50000 100000", 0), ("600000 100000", 6), ("max 100000", None), ("-1 100000", None), ("1 0", None), ("garbled", None)],
)
def test_cgroup_v2_fractional_unlimited_and_invalid_limits(monkeypatch, value, expected):
    mock_cgroup(monkeypatch, {"/sys/fs/cgroup/team/task/cpu.max": value})
    assert _workers._cgroup_cpu_count() == expected


@pytest.mark.parametrize("membership", ["0::/host/team/task", "0::/task"])
def test_cgroup_mount_root_and_namespace_paths(monkeypatch, membership):
    mock_cgroup(
        monkeypatch, {"/limits cpu/task/cpu.max": "300000 100000"}, membership=membership, mount="/host/team", location="/limits cpu"
    )
    assert _workers._cgroup_cpu_count() == 3


def test_cgroup_v1_cpu_controller_and_parent_limit(monkeypatch):
    mock_cgroup(
        monkeypatch,
        {
            "/sys/fs/cgroup/team/task/cpu.cfs_quota_us": "-1",
            "/sys/fs/cgroup/team/task/cpu.cfs_period_us": "100000",
            "/sys/fs/cgroup/team/cpu.cfs_quota_us": "800000",
            "/sys/fs/cgroup/team/cpu.cfs_period_us": "100000",
        },
        membership="4:cpu,cpuacct:/team/task",
        version=1,
    )
    assert _workers._cgroup_cpu_count() == 8


def test_missing_or_malformed_cgroup_metadata_is_ignored(monkeypatch):
    monkeypatch.setattr(_workers, "_read_text", lambda _: "invalid metadata")
    assert _workers._cgroup_cpu_count() is None


def test_unreadable_quota_is_ignored(tmp_path):
    assert _workers._read_text(tmp_path / "missing") == ""
    assert _workers._read_text(tmp_path) == ""


@pytest.mark.parametrize(
    "quota,slurm,expected",
    [(2, "64", 2), (32, "6", 6), (None, "7", 7), (None, "garbled", 64), (None, "0", 64), (None, "-1", 64), (128, "128", 64), (0, "8", 0)],
)
def test_available_cpus_use_smallest_valid_budget(monkeypatch, quota, slurm, expected):
    monkeypatch.setattr(_workers.os, "process_cpu_count", lambda: 128, raising=False)
    monkeypatch.setattr(_workers.os, "sched_getaffinity", lambda _: set(range(64)), raising=False)
    monkeypatch.setattr(_workers, "_cgroup_cpu_count", lambda: quota)
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", slurm)
    assert _workers._available_cpu_count() == expected
    assert _workers._default_worker_count(32) <= max(0, expected - 4)
