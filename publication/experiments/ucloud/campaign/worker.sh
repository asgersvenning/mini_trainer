#!/usr/bin/env bash
# GPU worker: drain every campaign study (prepare, qualify, train, predict) until no free work remains.
set -euo pipefail
revision="$1"
base="/work/results/campaign-${revision:0:7}"
mkdir -p "$base/logs"
exec > >(tee -a "$base/logs/worker-$(hostname)-$MT_STARTED.log") 2>&1
source /work/mini_trainer/publication/experiments/training_ablations/setup.sh
WANDB_API_KEY="$(< /work/mini-trainer-secrets/wandb-api-key)"
export WANDB_API_KEY
nvidia-smi
# Leave a quarter hour of the allocation for cleanup.
hours="$(awk -v a="$MT_ALLOCATION_HOURS" -v s="$MT_STARTED" -v n="$(date +%s)" 'BEGIN { print a - (n - s) / 3600 - 0.25 }')"
python -m publication.experiments.training_ablations.campaign work "$base" --hours "$hours"
