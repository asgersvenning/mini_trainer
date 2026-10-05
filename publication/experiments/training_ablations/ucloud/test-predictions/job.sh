#!/usr/bin/env bash
# MT_STAGE=predict (GPU node): predict the test split of completed ablation studies, export one
# evidence snapshot and upload it. MT_STAGE=bootstrap (CPU node): run the mini_metrics bootstrap
# of that snapshot on all cores and upload it as a separate replicates folder. Uploads go to ERDA
# with the dedicated key from the secrets mount unless MT_UPLOAD=0.
# Arguments: REVISION NAME=STUDY_ROOT [NAME=STUDY_ROOT ...]. Reruns resume finished work.
set -euo pipefail
revision="$1"
shift
stage="${MT_STAGE:?Set MT_STAGE=predict or MT_STAGE=bootstrap}"
root="/work/results/test-predictions-${revision:0:7}"
mkdir -p "$root/cohorts"
# Keep the first run's snapshot name so resumed and bootstrap jobs use the same snapshot.
[[ -e "$root/snapshot-name" ]] || echo "evidence-$(date +%Y%m%d)-${revision:0:7}" > "$root/snapshot-name"
snapshot="$(< "$root/snapshot-name")"
exec > >(tee -a "$root/$stage.log") 2>&1
trap 'printf "%s\n" "$?" > "$root/$stage.exit-code"' EXIT
source /work/mini_trainer/publication/experiments/training_ablations/setup.sh

trap 'code=$?; rm -f "${erda_key:-}"; printf "%s\n" "$code" > "$root/$stage.exit-code"' EXIT
source /work/mini_trainer/publication/experiments/ucloud/erda.sh

cohorts=()
for spec in "$@"; do
    cohorts+=("$root/cohorts/${spec%%=*}")
done

if [[ "$stage" == predict ]]; then
    nvidia-smi
    for spec in "$@"; do
        cohort="$root/cohorts/${spec%%=*}"
        mkdir -p "$cohort"
        ln -sfn "${spec#*=}" "$cohort/study"
        python -m publication.experiments.training_ablations.study predict "${spec#*=}" --output "$cohort/predictions"
    done
    if [[ ! -e "$root/$snapshot/manifest.json" ]]; then
        rm -rf "${root:?}/$snapshot"
        python -m publication.experiments.evidence "$root/$snapshot" "${cohorts[@]}" \
            --source-metadata /work/plantnet/plantnet300K_metadata.csv
    fi
    python -m publication.experiments.artifacts verify "$root/$snapshot/manifest.json" "$root/$snapshot"
    upload "$root/$snapshot"
elif [[ "$stage" == bootstrap ]]; then
    python -m publication.experiments.artifacts verify "$root/$snapshot/manifest.json" "$root/$snapshot"
    # The metrics dependency group pins mini_metrics.
    uv sync --project /work/mini_trainer --python 3.13 --frozen --extra recommended \
        --extra "${MT_TORCH_BACKEND:-cu130}" --group metrics
    replicates="$root/replicates-${snapshot#evidence-}"
    mkdir -p "$replicates"
    for cohort in "${cohorts[@]}"; do
        out="$replicates/$(basename "$cohort")"
        [[ -e "$out/resampling.json" ]] && continue
        rm -rf "$out"
        PYTHONHASHSEED=0 python -m publication.experiments.statistics.replicates "$root/$snapshot" "$(basename "$cohort")" \
            "$out" --replicates "${MT_REPLICATES:-1000}" --workers "$(( $(nproc) - 4 ))" --chunk 10
    done
    if [[ ! -e "$replicates/manifest.json" ]]; then
        (cd "$replicates" && find . -type f | sed 's#^\./##' | sort > "$root/replicate-files.txt")
        python -m publication.experiments.artifacts create "$replicates" "$replicates/manifest.json" \
            --files "$root/replicate-files.txt" --revision "$revision"
    fi
    upload "$replicates"
else
    echo "Unknown MT_STAGE: $stage" >&2
    exit 2
fi
