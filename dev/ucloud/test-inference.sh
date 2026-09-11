#!/usr/bin/env bash
# Stage the complete saved test split and infer without changing the environment.
set -euo pipefail
repo=$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
output="${1:-/work/test-full-1}"
if (( $# > 0 )); then shift; fi
exec bash "$repo/dev/ucloud/expert-trial.sh" "$output" \
    --source /work/global_lepi \
    --data-index /work/results/global_lepi_production_w32_1/data_index.json \
    --copy-workers 512 --max-mib 131072 \
    --stage-timeout 3600 --inference-timeout 3600 --gpu 0 "$@"
