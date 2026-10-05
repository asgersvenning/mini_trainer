#!/usr/bin/env bash
# Predict the test split of completed ablation studies, export one evidence snapshot and,
# unless MT_UPLOAD=0, upload it to ERDA with the dedicated key from the secrets mount.
# Arguments: REVISION NAME=STUDY_ROOT [NAME=STUDY_ROOT ...]. Reruns resume finished runs.
set -euo pipefail
revision="$1"
shift
root="/work/results/test-predictions-${revision:0:7}"
snapshot="evidence-$(date +%Y%m%d)-${revision:0:7}"
mkdir -p "$root/cohorts"
exec > >(tee -a "$root/job.log") 2>&1
trap 'printf "%s\n" "$?" > "$root/exit-code"' EXIT
source /work/mini_trainer/publication/experiments/training_ablations/setup.sh
nvidia-smi

cohorts=()
for spec in "$@"; do
    name="${spec%%=*}" study="${spec#*=}"
    cohort="$root/cohorts/$name"
    mkdir -p "$cohort"
    ln -sfn "$study" "$cohort/study"
    python -m publication.experiments.training_ablations.study predict "$study" --output "$cohort/predictions"
    cohorts+=("$cohort")
done

[[ ! -e "$root/$snapshot" ]] || { echo "Snapshot already exists: $root/$snapshot" >&2; exit 2; }
python -m publication.experiments.evidence "$root/$snapshot" "${cohorts[@]}" \
    --source-metadata /work/plantnet/plantnet300K_metadata.csv
python -m publication.experiments.artifacts verify "$root/$snapshot/manifest.json" "$root/$snapshot"
[[ "${MT_UPLOAD:-1}" == 1 ]] || { echo "Upload skipped: $root/$snapshot"; exit 0; }

key="$(mktemp)"
trap 'code=$?; rm -f "$key"; printf "%s\n" "$code" > "$root/exit-code"' EXIT
install -m 600 /work/mini-trainer-secrets/erda-upload-key "$key"
erda=(sftp -F /dev/null -i "$key" -o IdentitiesOnly=yes -o IdentityAgent=none -o BatchMode=yes
    -o StrictHostKeyChecking=yes -o UserKnownHostsFile=/work/mini-trainer-secrets/erda-known-hosts
    -o GlobalKnownHostsFile=/dev/null -P 2222 -b - asgersvenning@ecos.au.dk@io.erda.au.dk)
remote="/publications/hierarchical_classification/$snapshot"
printf 'mkdir %s\nput -r %s/* %s/\n' "$remote" "$root/$snapshot" "$remote" | "${erda[@]}"
printf 'get %s/manifest.json %s/uploaded-manifest.json\n' "$remote" "$root" | "${erda[@]}"
cmp "$root/uploaded-manifest.json" "$root/$snapshot/manifest.json"
echo "Uploaded $remote"
