#!/usr/bin/env bash
# Export the head weights of every current run (the ablation campaign and the Gefion studies) as an
# evidence-format snapshot and upload it to ERDA. Each checkpoint is checked against its training
# record. Reruns resume.
set -euo pipefail
revision="$1"
root="/work/results/heads-${revision:0:7}"
mkdir -p "$root"
[[ -e "$root/snapshot-name" ]] || echo "heads-$(date +%Y%m%d)-${revision:0:7}" > "$root/snapshot-name"
snapshot="$root/$(< "$root/snapshot-name")"
exec > >(tee -a "$root/job.log") 2>&1
trap 'code=$?; rm -f "${erda_key:-}"; printf "%s\n" "$code" > "$root/exit-code"' EXIT
source /work/mini_trainer/publication/experiments/training_ablations/setup.sh
source /work/mini_trainer/publication/experiments/ucloud/erda.sh

# Gefion Flemming reuses the Global Lepidoptera models; its tables carry its own study name.
cohorts=(
    /work/results/campaign-808f65e/{lepi-512,lepi-512-duration,lepi-1513,lepi-1513-capped,plantnet}
    /work/results/gefion-eae9293/cohorts/{gefion-global-lepi,gefion-plantnet}
    /work/results/gefion-flemming-2726692/cohorts/gefion-flemming
)
if [[ ! -e "$snapshot/manifest.json" ]]; then
    rm -rf "$snapshot"
    python -m publication.experiments.heads export "$snapshot" "${cohorts[@]}" --revision "$revision"
fi
python -m publication.experiments.artifacts verify "$snapshot/manifest.json" "$snapshot"
upload "$snapshot"
