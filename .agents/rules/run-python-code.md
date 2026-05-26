---
trigger: always_on
---

**All** Python code must be run via the virtual environment managed by `uv`.
**Any** time you want to use `uv run` or other `uv` commands that could trigger `uv sync`, you should add the flag `--no-sync`.
If you want to run `uv sync`, do so explicitly, and make sure to follow the same tip I have written in the README for users of the package:

> [!TIP]
> We highly recommend installing `torch` and `torchvision` with native CUDA support via either `uv sync ... --extra [cpu/cu126/cu130/cu132]` or `uv pip install ... --torch-backend=auto`, **and** crucially running scripts or tools associated with your `uv` virtual environment by **activating the venv:**
> ```bash
> source .venv/bin/activate
> ```
> Using `uv run ...` is likely to automatically install CUDA-incompatible wheels. If you really want to use `uv run`, we suggest using the `--no-sync` flag every time.
> Note that if you are *"lucky"* you might have the default CUDA version on your system, meaning that `uv run` might in fact use the correct wheels. This is, however, not guaranteed.
