#!/usr/bin/env bash
# Archive the final checkpoints (last.pt) not yet on ERDA: the factorial, overnight and duration
# ablation studies and the Gefion backbone x head runs, in the study-relative layout of the
# earlier archive, with a catalog (study, run_id, sha256) and manifest. Every copy must match the
# hash its training record states. Reruns resume.
set -euo pipefail
revision="$1"
root="/work/results/checkpoints-${revision:0:7}"
mkdir -p "$root"
[[ -e "$root/snapshot-name" ]] || echo "checkpoints-$(date +%Y%m%d)-${revision:0:7}" > "$root/snapshot-name"
snapshot="$root/$(< "$root/snapshot-name")"
exec > >(tee -a "$root/job.log") 2>&1
trap 'code=$?; rm -f "${erda_key:-}"; printf "%s\n" "$code" > "$root/exit-code"' EXIT
source /work/mini_trainer/publication/experiments/training_ablations/setup.sh
source /work/mini_trainer/publication/experiments/ucloud/erda.sh

if [[ ! -e "$snapshot/manifest.json" ]]; then
    python - "$snapshot" <<'EOF'
import json, shutil, sys
from pathlib import Path

import pandas as pd

from publication.experiments.training_ablations.data import digest

snapshot = Path(sys.argv[1])
ablations = {
    "factorial-11": Path("/work/results/factorial-11/study"),
    "overnight-12": Path("/work/results/overnight-12/study"),
    "duration-12": Path("/work/results/overnight-12/duration"),
}
rows = []


def archive(source, relative, study, run_id, expected):
    target = snapshot / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        shutil.copyfile(source, target)
    sha = digest(target)
    if sha != expected:
        raise ValueError(f"Checkpoint hash differs from its training record: {source}")
    rows.append({"path": relative, "study": study, "run_id": run_id, "sha256": sha, "bytes": target.stat().st_size, "source": str(source)})


for study, root in ablations.items():
    for run in json.loads((root / "plan.json").read_text()):
        attempt = sorted((root / "runs" / run["id"]).glob("attempt-*"))[-1]
        record = json.loads((attempt / "train.json").read_text())
        relative = f"{study}/study/runs/{run['id']}/{attempt.name}/model/weights/last.pt"
        archive(attempt / "model/weights/last.pt", relative, study, run["id"], record["weights_sha256"])

# Gefion runs come from the Gefion evidence studies, which record each run's checkpoint and hash.
# Flemming reuses the Global Lepidoptera models.
gefion = Path("/work/results/gefion-eae9293/cohorts")
for study in ["gefion-global-lepi", "gefion-plantnet"]:
    root = gefion / study / "study"
    for run in json.loads((root / "plan.json").read_text()):
        record = json.loads((root / "runs" / run["id"] / "attempt-000" / "train.json").read_text())
        relative = f"{study}/study/runs/{run['id']}/attempt-000/model/weights/last.pt"
        archive(Path(record["weights"]), relative, study, run["id"], record["weights_sha256"])

catalog = pd.DataFrame(rows).sort_values(["study", "run_id"])
catalog.to_csv(snapshot / "catalog.csv", index=False)
(snapshot / "README.md").write_text(
    "# Final checkpoints\n\nFinal-epoch weights (last.pt) behind the evidence snapshots, at study-relative paths.\n"
    "`catalog.csv` maps each file to `study` and `run_id`; its `sha256` equals the run's recorded `weights_sha256`.\n"
)
print(catalog.groupby("study").size().to_dict(), f"{catalog.bytes.sum() / 1e9:.1f} GB")
EOF
    (cd "$snapshot" && find . -type f ! -name manifest.json | sed 's#^\./##' | sort > "$root/files.txt")
    python -m publication.experiments.artifacts create "$snapshot" "$snapshot/manifest.json" --files "$root/files.txt" --revision "$revision"
fi
python -m publication.experiments.artifacts verify "$snapshot/manifest.json" "$snapshot"
upload "$snapshot"
