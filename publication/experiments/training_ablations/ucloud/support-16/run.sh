#!/usr/bin/env bash
set +x
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
index="${1:?Specify node index}"
revision="${2:?Specify reviewed repository commit}"
started="$(date +%s)"
root=/work/results/support-16
mkdir -p "$root"
exec > >(tee -a "$root/node-${index}.log") 2>&1
record_exit() { code=$?; printf '%s\n' "$code" > "$root/node-${index}.exit-code"; }
trap record_exit EXIT

curl -fLsS https://astral.sh/uv/install.sh | sh
git clone https://github.com/asgersvenning/mini_trainer.git /work/mini_trainer
git -C /work/mini_trainer checkout --detach "$revision"
source /work/mini_trainer/publication/experiments/training_ablations/setup.sh
export WANDB_API_KEY="$(< /work/mini-trainer-secrets/wandb-api-key)"
python /work/support-bootstrap/coordinate.py "$index" "$started"
