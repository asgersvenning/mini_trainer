#!/usr/bin/env bash
# Rebuild the Gefion Flemming study with the reviewed species-key corrections, reuse its existing
# predictions (labels do not affect model outputs), export a self-contained snapshot and upload it.
set -euo pipefail
revision="$1"
source_root="${MT_GEFION_ROOT:-/work/results/gefion-eae9293}"
root="/work/results/gefion-flemming-${revision:0:7}"
mkdir -p "$root/cohorts"
[[ -e "$root/snapshot-name" ]] || echo "gefion-flemming-evidence-$(date +%Y%m%d)-${revision:0:7}" > "$root/snapshot-name"
snapshot="$(< "$root/snapshot-name")"
exec > >(tee -a "$root/job.log") 2>&1
trap 'code=$?; rm -f "${erda_key:-}"; printf "%s\n" "$code" > "$root/exit-code"' EXIT
source /work/mini_trainer/publication/experiments/training_ablations/setup.sh
source /work/mini_trainer/publication/experiments/ucloud/erda.sh

transfer=/work/gefion-transfer cohort="$root/cohorts/gefion-flemming"
runs=()
for campaign in "$transfer"/meta/gefion-main/*/results/runs; do
    for head in flat hierarchical conditional independent; do
        runs+=("$campaign/${head}_global_lepi")
    done
done
if [[ ! -e "$cohort/study/plan.json" ]]; then
    rm -rf "$cohort"
    python -m publication.experiments.gefion prepare "$cohort" --evaluation flemming --weights-root "$transfer" \
        --runs "${runs[@]}" --index "$transfer/global_lepi_flat_data_index.json" \
        --flemming-labels "$transfer/flemming_hierarchical_mini_metric.csv" \
        --corrections /work/mini_trainer/publication/experiments/flemming-corrections.csv
    ln -s "$source_root/cohorts/gefion-flemming/predictions" "$cohort/predictions"
fi
if [[ ! -e "$root/$snapshot/manifest.json" ]]; then
    rm -rf "${root:?}/$snapshot"
    python -m publication.experiments.evidence "$root/$snapshot" "$cohort"
fi
python -m publication.experiments.artifacts verify "$root/$snapshot/manifest.json" "$root/$snapshot"
upload "$root/$snapshot"
