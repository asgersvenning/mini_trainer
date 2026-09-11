#!/usr/bin/env bash
# CPU evaluation of completed prediction CSVs, independent of the training venv.
set -euo pipefail
selection="${1:-all}"
output="${2:-/work/evaluation-1}"
case "$selection" in
    expert) datasets=(expert) ;;
    test) datasets=(test) ;;
    all) datasets=(expert test) ;;
    *) echo 'Usage: evaluate-results.sh [expert|test|all] [fresh-output-directory]' >&2; exit 2 ;;
esac
revision=70cc69adc05362863439277048e06386c1f885e1
package="git+https://github.com/GuillaumeMougeot/mini_metrics.git@$revision"
expert_csv="${MT_EXPERT_CSV:-/work/expert-full-2/predictions/mini_metric.csv}"
test_csv="${MT_TEST_CSV:-/work/test-full-1/predictions/mini_metric.csv}"
for dataset in "${datasets[@]}"; do
    if [[ "$dataset" == expert ]]; then input="$expert_csv"; else input="$test_csv"; fi
    if [[ ! -s "$input" ]]; then
        echo "Missing or empty completed prediction file: $input. Finish inference first." >&2
        exit 1
    fi
    if [[ -e "$output/$dataset" ]]; then
        echo "Output already exists: $output/$dataset; choose a fresh output directory." >&2
        exit 1
    fi
done
export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg
runner=(uvx --python 3.13 --from "$package" mm_metrics)
# Resolve the isolated CPU tool once before starting reports.
"${runner[@]}" --help > /dev/null
for dataset in "${datasets[@]}"; do
    if [[ "$dataset" == expert ]]; then input="$expert_csv"; else input="$test_csv"; fi
    target="$output/$dataset"
    mkdir -p "$output"
    mkdir "$target"
    sha256sum "$input" > "$target/input.sha256"
    printf 'mini_metrics_revision=%s\ninput=%s\n' "$revision" "$input" > "$target/provenance.txt"
    for scope in all_labels known_labels per_class; do
        options=()
        case "$scope" in
            all_labels) options=(--all) ;;
            known_labels) options=(--all --known-only) ;;
            per_class) options=(--per-class) ;;
        esac
        command=("${runner[@]}" --files "$input" --output-dir "$target" --output-name "$scope" --precision 10 "${options[@]}")
        printf '%q ' "${command[@]}" >> "$target/commands.sh"
        printf '\n' >> "$target/commands.sh"
        echo "$dataset / $scope: $target/$scope.log"
        if ! "${command[@]}" 2>&1 | tee "$target/$scope.log"; then
            echo "Evaluation failed; inspect $target/$scope.log" >&2
            exit 1
        fi
    done
    touch "$target/COMPLETED"
    echo "Completed $dataset evaluation: $target"
done
