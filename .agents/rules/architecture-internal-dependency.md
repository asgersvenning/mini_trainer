---
trigger: always_on
---

# Architecture

Follow the compatibility and architecture section of [AGENTS.md](../../AGENTS.md).
Read the import contracts in `pyproject.toml` before changing imports.
Use `bash dev/check.sh static` to validate architecture without importing training code.
Use behavioral tests for runtime changes as described in [dev/README.md](../../dev/README.md).
