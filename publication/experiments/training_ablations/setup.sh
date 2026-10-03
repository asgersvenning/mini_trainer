#!/usr/bin/env bash
# Dedicated UCloud environment; deliberately select CUDA and retain the lock.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null; then
    curl -fLsS https://astral.sh/uv/install.sh | sh
fi
command -v uv >/dev/null || { echo 'uv installation did not put uv on PATH.' >&2; exit 1; }
MT_ABLATION_REPO="${MT_ABLATION_REPO:-/work/mini_trainer}"
MT_TORCH_BACKEND="${MT_TORCH_BACKEND:-cu130}"
case "$MT_TORCH_BACKEND" in cu126|cu130|cu132) ;; *) echo 'Choose cu126, cu130, or cu132' >&2; exit 1 ;; esac
export UV_PROJECT_ENVIRONMENT="${MT_ABLATION_ENV:-/work/venvs/mt-ablations}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/work/.cache/uv}"
export TORCH_HOME="${TORCH_HOME:-/work/.cache/torch}"
export MPLBACKEND=Agg
uv sync --project "$MT_ABLATION_REPO" --python 3.13 --frozen --extra recommended --extra "$MT_TORCH_BACKEND"
export PATH="$UV_PROJECT_ENVIRONMENT/bin:$PATH"
cd "$MT_ABLATION_REPO"
