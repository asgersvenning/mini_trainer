#!/usr/bin/env bash
# Uploaded bootstrap only; the node clones the actual, pinned research checkout.
set +x
set -euo pipefail
revision="${1:?Supply the full Git commit to run}"
pilot_root="${2:?Supply a fresh path under the persistent results mount}"
login_mode="${3:---wait-for-wandb}"
[[ "$revision" =~ ^[0-9a-f]{40}$ ]] || { echo 'Expected a full Git commit' >&2; exit 2; }
[[ "$login_mode" == --wait-for-wandb || "$login_mode" == --authenticated ]] || exit 2
if [[ -n "${MT_WANDB_API_KEY_FILE:-}" ]]; then
    [[ -r "$MT_WANDB_API_KEY_FILE" ]] || { echo 'W&B credential file is not readable' >&2; exit 2; }
    WANDB_API_KEY="$(< "$MT_WANDB_API_KEY_FILE")"
    [[ -n "$WANDB_API_KEY" ]] || { echo 'W&B credential file is empty' >&2; exit 2; }
    export WANDB_API_KEY
fi
[[ ! -e "$pilot_root" ]] || { echo 'Use a fresh pilot directory' >&2; exit 2; }
mkdir -p "$pilot_root"
export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
exec > >(tee -a "$pilot_root/bootstrap.log") 2>&1
trap 'code=$?; printf "%s\n" "$code" > "$pilot_root/exit-code"' EXIT
printf 'Starting pilot at %s\n' "$(date -Is)"
command -v uv >/dev/null || curl -fLsS https://astral.sh/uv/install.sh | sh
export MT_ABLATION_REPO="${MT_ABLATION_REPO:-/work/mini_trainer}"
git clone https://github.com/asgersvenning/mini_trainer.git "$MT_ABLATION_REPO"
git -C "$MT_ABLATION_REPO" checkout --detach "$revision"
source "$MT_ABLATION_REPO/publication/experiments/training_ablations/setup.sh"
nvidia-smi
python -m publication.experiments.training_ablations.study prepare "$pilot_root/study" \
    --config "${MT_ABLATION_CONFIG:-publication/experiments/training_ablations/config.json}"
if [[ "$login_mode" == --wait-for-wandb ]]; then
    printf '\nIn the web terminal run these commands after successful login:\n%s/bin/wandb login\ntouch %q/wandb-ready\n' "$UV_PROJECT_ENVIRONMENT" "$pilot_root"
    touch "$pilot_root/waiting-for-wandb"
    for ((attempt=0; attempt<1200; attempt++)); do
        [[ ! -f "$pilot_root/wandb-ready" ]] || break
        sleep 1
    done
    [[ -f "$pilot_root/wandb-ready" ]] || { echo 'W&B login wait expired after 20 minutes'; exit 2; }
fi
python -m publication.experiments.training_ablations.study qualify "$pilot_root/study" --hours 0.4
python -m publication.experiments.training_ablations.study status "$pilot_root/study"
printf 'Pilot complete at %s\n' "$(date -Is)"
