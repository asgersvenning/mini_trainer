#!/usr/bin/env bash
# Use an existing export environment and a disposable runtime-only environment.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export_python="${1:-.venv/bin/python}"
export_python="$(cd -- "$(dirname -- "$export_python")" && pwd)/$(basename -- "$export_python")"
smoke_dir="$(mktemp -d "${TMPDIR:-/tmp}/mini-trainer-onnx.XXXXXXXX")"
trap 'rm -rf -- "$smoke_dir"' EXIT
export OMP_NUM_THREADS=1
CUDA_VISIBLE_DEVICES="" "$export_python" -I dev/onnx_smoke.py produce "$smoke_dir"
runtime_version="$("$export_python" -I -c 'from importlib.metadata import version; print(version("onnxruntime"))')"
uv venv "$smoke_dir/runtime" --python "$export_python"
uv pip install --python "$smoke_dir/runtime/bin/python" "onnxruntime==$runtime_version"
cp dev/onnx_smoke.py "$smoke_dir/check.py"
cd -- "$smoke_dir"
"$smoke_dir/runtime/bin/python" -I check.py verify "$smoke_dir"
