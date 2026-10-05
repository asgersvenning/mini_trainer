#!/usr/bin/env bash
# Rerun the Gefion backbone x head checkpoints on their test sets (and the Global Lepidoptera
# models on Flemming) through the streaming collector, export one evidence snapshot and upload it.
# Inputs: /work/gefion-transfer (checkpoints, data indexes, run metadata, Flemming labels),
# /work/global_lepi, /work/plantnet and /work/flemming. Reruns resume finished work.
set -euo pipefail
revision="$1"
root="/work/results/gefion-${revision:0:7}"
mkdir -p "$root/cohorts"
[[ -e "$root/snapshot-name" ]] || echo "gefion-evidence-$(date +%Y%m%d)-${revision:0:7}" > "$root/snapshot-name"
snapshot="$(< "$root/snapshot-name")"
exec > >(tee -a "$root/job.log") 2>&1
trap 'code=$?; rm -f "${erda_key:-}"; printf "%s\n" "$code" > "$root/exit-code"' EXIT
source /work/mini_trainer/publication/experiments/training_ablations/setup.sh
source /work/mini_trainer/publication/experiments/ucloud/erda.sh
nvidia-smi

transfer=/work/gefion-transfer
prepare() {  # NAME EVALUATION TRAINING_DATASET [extra prepare arguments]
    local cohort="$root/cohorts/$1" evaluation="$2" dataset="$3" runs=() campaign head
    shift 3
    for campaign in "$transfer"/meta/gefion-main/*/results/runs; do
        for head in flat hierarchical conditional independent; do
            runs+=("$campaign/${head}_$dataset")
        done
    done
    if [[ ! -e "$cohort/study/plan.json" ]]; then
        rm -rf "$cohort/study"
        python -m publication.experiments.gefion prepare "$cohort" --evaluation "$evaluation" --weights-root "$transfer" \
            --runs "${runs[@]}" "$@"
    fi
    python -m publication.experiments.gefion predict "$cohort"
}
prepare gefion-global-lepi global_lepi global_lepi --index "$transfer/global_lepi_flat_data_index.json" \
    --parquet /work/global_lepi/0032836-250426092105405_processing_metadata_postprocessed_quality_filtered.parquet
prepare gefion-plantnet plantnet plantnet --index /work/plantnet/data_index.json
prepare gefion-flemming flemming global_lepi --index "$transfer/global_lepi_flat_data_index.json" \
    --flemming-labels "$transfer/flemming_hierarchical_mini_metric.csv"

if [[ ! -e "$root/$snapshot/manifest.json" ]]; then
    rm -rf "${root:?}/$snapshot"
    python -m publication.experiments.evidence "$root/$snapshot" "$root"/cohorts/gefion-{global-lepi,plantnet,flemming} \
        --source-metadata /work/plantnet/plantnet300K_metadata.csv
fi
python -m publication.experiments.artifacts verify "$root/$snapshot/manifest.json" "$root/$snapshot"
upload "$root/$snapshot"
