#!/usr/bin/env bash
# Synced bootstrap only: clone the reviewed revision and run its job script.
set -euo pipefail
revision="${1:?Supply the full Git commit to run}"
shift
[[ "$revision" =~ ^[0-9a-f]{40}$ ]] || { echo "Not a full commit hash: $revision" >&2; exit 2; }
export PATH="$HOME/.local/bin:$PATH" PYTHONUNBUFFERED=1 MPLBACKEND=Agg
git clone https://github.com/asgersvenning/mini_trainer.git /work/mini_trainer
git -C /work/mini_trainer checkout --detach "$revision"
exec bash /work/mini_trainer/publication/experiments/training_ablations/ucloud/test-predictions/job.sh "$revision" "$@"
