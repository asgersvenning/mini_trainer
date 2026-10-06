#!/usr/bin/env bash
# Synced bootstrap only: clone the reviewed revision and run its campaign job.
# Arguments: worker|finalize REVISION ALLOCATION_HOURS
set -euo pipefail
job="${1:?worker or finalize}" revision="${2:?Supply the full Git commit}" allocation="${3:?Allocation hours}"
[[ "$job" == worker || "$job" == finalize ]] || { echo "Unknown job: $job" >&2; exit 2; }
[[ "$revision" =~ ^[0-9a-f]{40}$ ]] || { echo "Not a full commit hash: $revision" >&2; exit 2; }
export MT_STARTED="$(date +%s)" MT_ALLOCATION_HOURS="$allocation"
export PATH="$HOME/.local/bin:$PATH" PYTHONUNBUFFERED=1 MPLBACKEND=Agg
git clone https://github.com/asgersvenning/mini_trainer.git /work/mini_trainer
git -C /work/mini_trainer checkout --detach "$revision"
exec bash "/work/mini_trainer/publication/experiments/ucloud/campaign/$job.sh" "$revision"
