#!/usr/bin/env bash
# Dedicated environments for a fresh manually allocated UCloud job; no training.
set -euo pipefail

if (( $# != 1 )); then
    echo 'Usage: bash dev/ucloud/setup.sh /work/DATASET/metadata.parquet' >&2
    exit 2
fi
for tool in uv git python3; do
    command -v "$tool" >/dev/null || { echo "Missing $tool; install it before setup" >&2; exit 2; }
done
[[ -f "$1" ]] || { echo "Parquet file not found: $1" >&2; exit 2; }
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(git -C "$script_dir" rev-parse --show-toplevel)
work_root=${MT_WORK_ROOT:-/work}
mkdir -p "$work_root/venvs"
work_root=$(cd -- "$work_root" && pwd)
node_config=${MT_CONFIG:-$work_root/qualification.json}
node_config=$(python3 -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "$node_config")
[[ ! -e "$node_config" ]] || { echo "Config already exists: $node_config; preserve it or choose a new MT_CONFIG" >&2; exit 2; }
mkdir -p -- "$(dirname -- "$node_config")"
backend=${MT_TORCH_BACKEND:-cu130}
case "$backend" in
    cu126|cu130|cu132) ;;
    *) echo 'MT_TORCH_BACKEND must be cu126, cu130 or cu132' >&2; exit 2 ;;
esac
template=${MT_TEMPLATE:-$script_dir/qualification.json}
master_sha=$(python3 -c 'import json,sys; envs=json.load(open(sys.argv[1]))["environments"]; print(envs.get("master", envs["quant"])["commit"])' "$template")
quant_sha=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["environments"]["quant"]["commit"])' "$template")
for sha in "$master_sha" "$quant_sha"; do
    git -C "$repo_root" cat-file -e "$sha^{commit}" || { echo "Missing pinned commit $sha; fetch full repository history" >&2; exit 2; }
done
repo_url=${MT_REPO_URL:-https://github.com/asgersvenning/mini_trainer.git}
export UV_CACHE_DIR=${UV_CACHE_DIR:-$work_root/.cache/uv}
export UV_LINK_MODE=copy
unset UV_TORCH_BACKEND

export_dir=$(mktemp -d)
trap 'rm -rf -- "$export_dir"' EXIT
git -C "$repo_root" archive "$quant_sha" pyproject.toml uv.lock | tar -xf - -C "$export_dir"
requirements="$work_root/requirements-mt.txt"
echo "Exporting pinned dependencies from $quant_sha ($backend)"
uv export --directory "$export_dir" --locked --no-dev --no-emit-project \
    --extra recommended --extra "$backend" --extra export --extra quantization \
    --output-file "$requirements" >/dev/null

mapfile -t branches < <(python3 -c 'import json,sys; print("\n".join(json.load(open(sys.argv[1]))["environments"]))' "$template")
for branch in "${branches[@]}"; do
    env_dir="$work_root/venvs/mt-$branch"
    if [[ ! -x "$env_dir/bin/python" ]]; then
        uv venv --python 3.12 "$env_dir"
    fi
    echo "Installing locked dependencies in $env_dir"
    # These are dedicated comparison environments. Sync removes leftover packages
    # when retrying a partially completed setup; it never touches the node's Python.
    uv pip sync --python "$env_dir/bin/python" --require-hashes \
        --index https://pypi.org/simple --default-index "https://download.pytorch.org/whl/$backend" \
        --index-strategy unsafe-first-match "$requirements"
    sha=$master_sha
    if [[ "$branch" == quant ]]; then sha=$quant_sha; fi
    uv pip install --python "$env_dir/bin/python" --no-deps "mini_trainer @ git+$repo_url@$sha"
    uv pip check --python "$env_dir/bin/python"
done

python3 - "$template" "$1" "$work_root" "$node_config" <<'PY'
import json
import sys
from pathlib import Path

template, parquet, work_root, destination = map(Path, sys.argv[1:])
config = json.loads(template.read_text())
config["parquet"] = str(parquet.resolve())
config["output"] = str(work_root / "results" / Path(config["output"]).name)
for branch, entry in config["environments"].items():
    entry["python"] = str(work_root / "venvs" / f"mt-{branch}" / "bin" / "python")
with destination.open("x") as handle:
    json.dump(config, handle, indent=2)
    handle.write("\n")
print(f"Setup complete. Node config: {destination}")
PY
printf 'Next: bash %q %q --stage plan\n' "$script_dir/launch.sh" "$node_config"
echo 'Then run prepare followed immediately by train; the 30-minute budget starts at prepare.'
