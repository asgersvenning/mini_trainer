#!/usr/bin/env bash
# Run the bounded expert trial without changing the installed training environment.
set -euo pipefail
repo=$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
revision=cba4ecd
python=/work/venvs/mt-quant/bin/python
source_dir=/work/flemming_helsing/restructured/valid/referenced
weights=/work/results/global_lepi_production_w32_1/weights/best.pt
output="${1:-/work/expert-staging-trial-1}"
if (( $# > 0 )); then shift; fi
# An isolated source overlay keeps the exact trained package dependencies intact.
git -C "$repo" cat-file -e "$revision:mini_trainer/data/metadata.py"
code=$(mktemp -d /work/mt-inference-code.XXXXXX)
git -C "$repo" archive "$revision" mini_trainer | tar -x -C "$code"
export PYTHONPATH="$code"
exec "$python" "$repo/dev/ucloud/expert_trial.py" \
    --source "$source_dir" --weights "$weights" --output "$output" "$@"
