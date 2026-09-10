"""Agent documents must never hide code changes or break existing CI triggers."""

import re
import subprocess
from pathlib import Path

import pytest
import yaml

from dev.ci_scope import agent_document, needs_checks


@pytest.mark.parametrize(
    ("path", "agent_only"),
    [
        ("AGENTS.md", True),
        (".agents/README.md", True),
        (".agents/notes/2026-09-10-topic.md", True),
        (".agents/skills/example/SKILL.md", True),
        (".agents/.gitignore", True),
        (".agents/skills/example/check.py", False),
        (".agents/config.json", False),
        ("docs/roadmap.md", False),
        ("README.md", False),
        ("mini_trainer/model.py", False),
        (".github/workflows/ci.yml", False),
        ("nested/AGENTS.md", False),
    ],
)
def test_scope_matches_push_filters(path, agent_only):
    root = Path(__file__).resolve().parents[2]
    assert agent_document(path) is agent_only
    for name in ("ci.yml", "benchmarks.yml"):
        workflow = yaml.load((root / ".github/workflows" / name).read_text(), Loader=yaml.BaseLoader)
        patterns = workflow["on"]["push"]["paths-ignore"]
        # Translate only the small glob subset used by these workflows.
        regexes = [re.escape(p).replace(r"\*\*/", "(?:.*/)?").replace(r"\*", "[^/]*") for p in patterns]
        assert any(re.fullmatch(pattern, path) for pattern in regexes) is agent_only
        assert "paths-ignore" not in workflow["on"]["pull_request"]
        for job in workflow["jobs"].values():
            if job is workflow["jobs"]["scope"]:
                continue
            assert job["needs"] == "scope"
            assert "always()" in job["if"]
            assert "needs.scope.outputs.run_checks != 'false'" in job["if"]


def test_real_pr_diff_keeps_mixed_changes_and_renames(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def git(*args):
        return subprocess.check_output(["git", *args], text=True).strip()

    git("init", "-q")
    git("config", "user.name", "CI test")
    git("config", "user.email", "ci@example.invalid")
    Path("code.py").write_text("print('original')\n")
    git("add", "code.py")
    git("commit", "-qm", "initial")
    base = git("rev-parse", "HEAD")
    Path(".agents").mkdir()
    Path(".agents/note.md").write_text("# Note\n")
    git("add", ".agents/note.md")
    git("commit", "-qm", "agent: note")
    notes = git("rev-parse", "HEAD")
    assert not needs_checks(base, notes)
    assert needs_checks(base, base)  # An empty diff is not proof of agent-only changes.
    assert needs_checks("missing", notes)
    assert needs_checks(None, None)  # Push/manual/scheduled events keep normal checks.
    Path("code.py").write_text("print('changed')\n")
    git("add", "code.py")
    git("commit", "-qm", "agent: misleading prefix cannot suppress code checks")
    mixed = git("rev-parse", "HEAD")
    assert needs_checks(base, mixed)
    git("mv", "code.py", ".agents/renamed.md")
    git("commit", "-qm", "rename")
    assert needs_checks(mixed, git("rev-parse", "HEAD"))


def test_benchmark_scheduled_and_manual_events_remain_available():
    root = Path(__file__).resolve().parents[2]
    workflow = yaml.load((root / ".github/workflows/benchmarks.yml").read_text(), Loader=yaml.BaseLoader)
    assert "schedule" in workflow["on"] and "workflow_dispatch" in workflow["on"]
    assert "ENABLE_GPU_BENCHMARKS" in workflow["jobs"]["gpu"]["if"]
    assert "inputs.gpu" in workflow["jobs"]["gpu"]["if"]
