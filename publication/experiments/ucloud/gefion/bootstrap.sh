#!/usr/bin/env bash
# mini_metrics bootstrap of the Gefion studies (CPU node), uploaded to ERDA as one replicates folder.
# Global Lepidoptera (633k test images) gets fewer replicates and workers; draws are sequential,
# so more can be added later. Flemming uses the label-corrected snapshot. Reruns resume per study.
set -euo pipefail
revision="$1"
root="/work/results/gefion-bootstrap-${revision:0:7}"
mkdir -p "$root"
[[ -e "$root/snapshot-name" ]] || echo "gefion-replicates-$(date +%Y%m%d)-${revision:0:7}" > "$root/snapshot-name"
replicates="$root/$(< "$root/snapshot-name")"
exec > >(tee -a "$root/job.log") 2>&1
trap 'code=$?; rm -f "${erda_key:-}"; printf "%s\n" "$code" > "$root/exit-code"' EXIT
source /work/mini_trainer/publication/experiments/training_ablations/setup.sh
source /work/mini_trainer/publication/experiments/ucloud/erda.sh
# The metrics dependency group pins mini_metrics.
uv sync --project /work/mini_trainer --python 3.13 --frozen --extra recommended \
    --extra "${MT_TORCH_BACKEND:-cu130}" --group metrics

gefion=/work/results/gefion-eae9293/gefion-evidence-20261005-eae9293
flemming=/work/results/gefion-flemming-2726692/gefion-flemming-evidence-20261006-2726692
cores=$(($(nproc) - 4))
# Study, snapshot, replicates, workers (each Global Lepidoptera worker needs about 6 GB).
studies=(
    "gefion-plantnet $gefion 1000 $cores"
    "gefion-flemming $flemming 1000 $cores"
    "gefion-global-lepi $gefion 200 40"
)
mkdir -p "$replicates"
for spec in "${studies[@]}"; do
    read -r study snapshot count workers <<< "$spec"
    out="$replicates/$study"
    [[ -e "$out/resampling.json" ]] && continue
    rm -rf "$out"
    PYTHONHASHSEED=0 python -m publication.experiments.statistics.replicates "$snapshot" "$study" "$out" \
        --replicates "$count" --workers "$workers" --chunk 10
done
if [[ ! -e "$replicates/manifest.json" ]]; then
    (cd "$replicates" && find . -type f | sed 's#^\./##' | sort > "$root/files.txt")
    python -m publication.experiments.artifacts create "$replicates" "$replicates/manifest.json" \
        --files "$root/files.txt" --revision "$revision"
fi
upload "$replicates"
