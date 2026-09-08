#!/usr/bin/env bash
# Explicit installation check in a disposable environment; leaves .venv untouched.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

smoke_python="${1:-.venv/bin/python}"
smoke_dir="$(mktemp -d "${TMPDIR:-/tmp}/mini-trainer-wheel.XXXXXXXX")"
trap 'rm -rf -- "$smoke_dir"' EXIT

uv build --wheel --out-dir "$smoke_dir/dist" --python "$smoke_python"
UV_PROJECT_ENVIRONMENT="$smoke_dir/env" uv sync --locked --no-dev --extra cpu --no-install-project --python "$smoke_python"
uv pip install --python "$smoke_dir/env/bin/python" --no-deps "$smoke_dir"/dist/*.whl
mkdir "$smoke_dir/smoke_fixture"
touch "$smoke_dir/smoke_fixture/__init__.py"
cp dev/wheel_smoke.py "$smoke_dir/smoke_fixture/check.py"
cd -- "$smoke_dir"
# Make the custom test backbone importable for weights-only reconstruction.
CUDA_VISIBLE_DEVICES="" MPLBACKEND=Agg MPLCONFIGDIR="$smoke_dir/matplotlib" "$smoke_dir/env/bin/python" -I -c \
    'import sys; sys.path.insert(0, "."); from smoke_fixture.check import main; main()'
