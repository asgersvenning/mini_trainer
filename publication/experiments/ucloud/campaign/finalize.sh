#!/usr/bin/env bash
# CPU finalizer: once every run is complete and predicted, export the evidence snapshot, bootstrap
# it with mini_metrics, archive the complete study directories and upload all three to ERDA.
# Reruns resume; uploads are skipped with MT_UPLOAD=0.
set -euo pipefail
revision="$1"
base="/work/results/campaign-${revision:0:7}"
mkdir -p "$base/logs"
exec > >(tee -a "$base/logs/finalize-$MT_STARTED.log") 2>&1
source /work/mini_trainer/publication/experiments/training_ablations/setup.sh
trap 'rm -f "${erda_key:-}"' EXIT
source /work/mini_trainer/publication/experiments/ucloud/erda.sh
python -m publication.experiments.training_ablations.campaign check "$base"

# Keep the first attempt's date so reruns finish the same folders.
[[ -e "$base/snapshot-date" ]] || date +%Y%m%d > "$base/snapshot-date"
suffix="$(< "$base/snapshot-date")-${revision:0:7}"
evidence="$base/ablation-evidence-$suffix" replicates="$base/ablation-replicates-$suffix" archive="$base/ablation-runs-$suffix"
read -ra studies <<< "$(python -c 'from publication.experiments.training_ablations.campaign import STUDIES; print(*STUDIES)')"

if [[ ! -e "$evidence/manifest.json" ]]; then
    rm -rf "$evidence"
    python -m publication.experiments.evidence "$evidence" "${studies[@]/#/$base/}" \
        --source-metadata /work/plantnet/plantnet300K_metadata.csv
fi
python -m publication.experiments.artifacts verify "$evidence/manifest.json" "$evidence"

# The metrics dependency group pins mini_metrics.
uv sync --project /work/mini_trainer --python 3.13 --frozen --extra recommended \
    --extra "${MT_TORCH_BACKEND:-cu130}" --group metrics
mkdir -p "$replicates"
for name in "${studies[@]}"; do
    [[ -e "$replicates/$name/resampling.json" ]] && continue
    rm -rf "${replicates:?}/$name"
    PYTHONHASHSEED=0 python -m publication.experiments.statistics.replicates "$evidence" "$name" "$replicates/$name" \
        --replicates 1000 --workers "$(($(nproc) - 4))" --chunk 10
done

# The raw record behind the evidence: configs, plans, logs, metrics, weights and test predictions
# with their provenance. Hard links avoid copying; locks are runtime state.
if [[ ! -e "$archive/manifest.json" ]]; then
    rm -rf "$archive"
    mkdir -p "$archive"
    for name in "${studies[@]}"; do
        cp -al "$base/$name" "$archive/$name"
    done
    find "$archive" -name '*.lock' -delete
fi
for folder in "$replicates" "$archive"; do
    if [[ ! -e "$folder/manifest.json" ]]; then
        (cd "$folder" && find . -type f ! -name manifest.json | sed 's#^\./##' | sort > "$folder.files")
        python -m publication.experiments.artifacts create "$folder" "$folder/manifest.json" --files "$folder.files" --revision "$revision"
    fi
done
for folder in "$evidence" "$replicates" "$archive"; do
    upload "$folder"
done
