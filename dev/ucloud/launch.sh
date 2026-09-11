#!/usr/bin/env bash
# Run once inside an allocated UCloud GPU job (one node, multiple GPUs).
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"
export MPLBACKEND=Agg
export PYTHONHASHSEED=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# Use the fresh training environment even when the node's system Python is older.
if (( $# == 0 )); then
    echo 'Usage: bash launch.sh comparison.json [--stage plan|prepare|train|summary] [--only RUN]' >&2
    exit 2
fi
controller_python=${CONTROLLER_PYTHON:-}
if [[ -z "$controller_python" ]]; then
    controller_python=$(python3 -c 'import json, os, sys; print(os.path.expanduser(json.load(open(sys.argv[1]))["environments"]["quant"]["python"]))' "$1")
fi
exec "$controller_python" "$script_dir/compare.py" "$@"
