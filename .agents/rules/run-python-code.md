---
trigger: always_on
---

# Python environment

Follow the environment and validation section of [AGENTS.md](../../AGENTS.md).
Use the existing `.venv` directly or `uv run --no-sync` for validation.
Installation is explicit: select a PyTorch backend and follow [dev/README.md](../../dev/README.md).
The installed-wheel check uses a disposable CPU environment and does not alter `.venv`.
