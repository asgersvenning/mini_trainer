"""Select code checks for PRs; unreadable or empty diffs conservatively run them."""

import os
import subprocess


def agent_document(path):
    return path in {"AGENTS.md", ".agents/.gitignore"} or (path.startswith(".agents/") and path.endswith(".md"))


def needs_checks(base, head):
    if not base or not head:
        return True
    try:
        changed = subprocess.check_output(
            ["git", "diff", "--name-only", "--no-renames", "-z", f"{base}...{head}", "--"], stderr=subprocess.PIPE
        )
    except subprocess.CalledProcessError:
        return True
    paths = [os.fsdecode(path) for path in changed.split(b"\0") if path]
    return not paths or any(not agent_document(path) for path in paths)


if __name__ == "__main__":
    print(f"run_checks={str(needs_checks(os.environ.get('BASE_SHA'), os.environ.get('HEAD_SHA'))).lower()}")
