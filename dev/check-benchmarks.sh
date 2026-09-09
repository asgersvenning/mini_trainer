#!/usr/bin/env bash
# Shared pipeline entry point; uses the selected environment without synchronizing it.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
mode="${1:-cpu}"
results="${2:?Supply a new results directory}"
benchmark_python="${BENCHMARK_PYTHON:-.venv/bin/python}"
if [[ -e "$results" ]]; then
    echo 'Results directory must be new.' >&2
    exit 2
fi
case "$mode" in cpu|gpu|real|qt|qt-real|qt-dense|qt-large-batch|qt-cudagraphs|qt-optimizer-cudagraphs|qt-efficientnet) ;; *) echo 'Mode must be cpu, gpu, real, qt, qt-real, qt-dense, qt-large-batch, qt-cudagraphs, qt-optimizer-cudagraphs or qt-efficientnet.' >&2; exit 2 ;; esac
if [[ "$mode" == qt-efficientnet ]]; then
    : "${BENCHMARK_DATA_ROOT:?Set BENCHMARK_DATA_ROOT to the directory containing blair/}"
    : "${BLAIR_CLASS_SPEC:?Set BLAIR_CLASS_SPEC to a reviewed Blair class specification}"
    : "${BENCHMARK_METRICS_PYTHON:?Set BENCHMARK_METRICS_PYTHON to an environment with mini_metrics installed}"
    epochs="${BENCHMARK_EPOCHS:-5}"
    [[ "$epochs" =~ ^[1-9][0-9]*$ ]] || { echo 'BENCHMARK_EPOCHS must be a positive integer.' >&2; exit 2; }
    read -r -a seeds <<< "${BENCHMARK_SEEDS:-42 43 44}"
    [[ "${#seeds[@]}" -gt 0 ]] || { echo 'BENCHMARK_SEEDS must not be empty.' >&2; exit 2; }
    declare -A seen_seeds=()
    for seed in "${seeds[@]}"; do
        [[ "$seed" =~ ^(0|[1-9][0-9]*)$ ]] || { echo 'BENCHMARK_SEEDS must contain nonnegative integers.' >&2; exit 2; }
        [[ -z "${seen_seeds[$seed]:-}" ]] || { echo 'BENCHMARK_SEEDS must be unique.' >&2; exit 2; }
        seen_seeds[$seed]=1
    done
    case "${BENCHMARK_HEAD:-both}" in
        both) heads=(flat hierarchical) ;;
        flat|hierarchical) heads=("$BENCHMARK_HEAD") ;;
        *) echo 'BENCHMARK_HEAD must be both, flat or hierarchical.' >&2; exit 2 ;;
    esac
    case "${BENCHMARK_TRAINING_MODE:-both}" in
        both) training_modes=(full frozen) ;;
        full|frozen) training_modes=("$BENCHMARK_TRAINING_MODE") ;;
        *) echo 'BENCHMARK_TRAINING_MODE must be both, full or frozen.' >&2; exit 2 ;;
    esac
    "$BENCHMARK_METRICS_PYTHON" -c 'import mini_metrics' # Fail before training if the optional environment is unavailable.
fi
mkdir -p -- "$results"
status=0
run_profile() {
    local profile="$1"
    shift
    if OMP_NUM_THREADS=1 MPLBACKEND=Agg TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}" \
        "$benchmark_python" -m dev.benchmarks.run --output "$results/$profile" "$@" > "$results/$profile.log" 2>&1; then
        return
    else
        local exit_code="$?"
        echo "Benchmark failed: $profile; see $results/$profile.log" >&2
        status=1
        if [[ ! -f "$results/$profile/report.json" ]]; then
            "$benchmark_python" - "$results/$profile/report.json" "$exit_code" "$@" <<'PY_REPORT'
import json
import sys
from pathlib import Path
path = Path(sys.argv[1])
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps({
    "schema_version": 1, "status": "failed", "arguments": sys.argv[3:],
    "quantized_training": "--quantized-training" in sys.argv[3:],
    "error": {"type": "ProcessFailure", "exit_code": int(sys.argv[2]), "message": "Process exited without a report; see profile log."},
}, indent=2) + "\n")
PY_REPORT
        fi
    fi
}
if [[ "$mode" == cpu ]]; then
    run_profile synthetic-cpu --device cpu
elif [[ "$mode" == gpu ]]; then
    for precision in float32 float16 bfloat16; do
        run_profile "synthetic-cuda-$precision" --device cuda:0 --dtype "$precision" --cache CUDA
    done
elif [[ "$mode" == qt ]]; then
    run_profile synthetic-float --device cuda:0 --dtype float16 --cache CPU --cache-workers 0
    run_profile synthetic-int8 --device cuda:0 --dtype float16 --cache CPU --cache-workers 0 --quantized-training
elif [[ "$mode" == qt-dense ]]; then
    : "${BENCHMARK_DATA_ROOT:?Set BENCHMARK_DATA_ROOT to the directory containing mnist/}"
    for precision in float int8; do
        quantization=()
        if [[ "$precision" == int8 ]]; then quantization=(--quantized-training); fi
        run_profile "mnist-dense-$precision" --dataset mnist --data-root "$BENCHMARK_DATA_ROOT/mnist" \
            --model-profile dense --optimizer sgd --learning-rate 0.3 --epochs 15 --batch-size 128 --compile \
            --device cuda:0 --dtype float16 --cache CPU --cache-workers 0 --allow-nondeterministic "${quantization[@]}"
    done
elif [[ "$mode" == qt-large-batch || "$mode" == qt-cudagraphs || "$mode" == qt-optimizer-cudagraphs ]]; then
    compile_mode=()
    profile=mnist-large-batch
    if [[ "$mode" == qt-cudagraphs || "$mode" == qt-optimizer-cudagraphs ]]; then
        compile_mode=(--compile-mode reduce-overhead)
        profile=mnist-cudagraphs
    fi
    if [[ "$mode" == qt-optimizer-cudagraphs ]]; then
        compile_mode+=(--optimizer-cudagraphs)
        profile=mnist-optimizer-cudagraphs
    fi
    : "${BENCHMARK_DATA_ROOT:?Set BENCHMARK_DATA_ROOT to the directory containing mnist/}"
    for seed in 42 43 44; do
        precisions=(float int8)
        if (( seed % 2 )); then precisions=(int8 float); fi
        for precision in "${precisions[@]}"; do
            quantization=()
            if [[ "$precision" == int8 ]]; then quantization=(--quantized-training); fi
            run_profile "$profile-$precision-seed$seed" --dataset mnist --data-root "$BENCHMARK_DATA_ROOT/mnist" \
                --seed "$seed" --model-profile dense --optimizer sgd --learning-rate 0.3 --epochs 60 --batch-size 512 \
                --compile "${compile_mode[@]}" --compile-optimizer --device cuda:0 --dtype float16 --cache CPU --cache-workers 0 \
                --allow-nondeterministic "${quantization[@]}"
        done
    done
elif [[ "$mode" == qt-efficientnet ]]; then
    for seed in "${seeds[@]}"; do
        precisions=(float int8)
        modes=("${training_modes[@]}")
        if [[ "$seed" =~ [13579]$ ]]; then
            precisions=(int8 float)
            if [[ "${#modes[@]}" == 2 ]]; then modes=(frozen full); fi
        fi
        for head in "${heads[@]}"; do
            for training_mode in "${modes[@]}"; do
                extra=()
                if [[ "$training_mode" == frozen ]]; then extra=(--fine-tune); fi
                for precision in "${precisions[@]}"; do
                    quantization=()
                    if [[ "$precision" == int8 ]]; then quantization=(--quantized-training); fi
                    run_profile "blair-$head-$training_mode-$precision-seed$seed" \
                        --dataset blair --data-root "$BENCHMARK_DATA_ROOT/blair" --class-spec "$BLAIR_CLASS_SPEC" \
                        --backbone efficientnet_v2_s --head "$head" --hidden symmetric --normalized --pretrained \
                        --seed "$seed" --epochs "$epochs" --batch-size 32 --image-size 128 --optimizer muon --learning-rate 0.01 \
                        --device cuda:0 --dtype bfloat16 --cache CPU --cache-workers 0 --num-workers 0 --threads 1 \
                        "${extra[@]}" "${quantization[@]}"
                done
            done
        done
    done
    # Evaluate only after all training timings, in the explicitly selected metrics environment.
    for seed in "${seeds[@]}"; do
        for head in "${heads[@]}"; do
            for training_mode in "${training_modes[@]}"; do
                pair="blair-$head-$training_mode-seed$seed"
                if "$benchmark_python" -m dev.benchmarks.training_predictions \
                    --baseline "$results/blair-$head-$training_mode-float-seed$seed" \
                    --candidate "$results/blair-$head-$training_mode-int8-seed$seed" \
                    --output "$results/$pair-inputs" > "$results/$pair-quality.log" 2>&1 && \
                    PYTHONHASHSEED=0 OMP_NUM_THREADS=1 "$BENCHMARK_METRICS_PYTHON" -m dev.benchmarks.quality_compare \
                    --manifest "$results/$pair-inputs/manifest.json" --output "$results/$pair-quality" \
                    >> "$results/$pair-quality.log" 2>&1; then
                    :
                else
                    echo "Quality evaluation failed: $pair; see $results/$pair-quality.log" >&2
                    status=1
                    "$benchmark_python" - "$results/$pair-quality/report.json" <<'PY_QUALITY_FAILURE'
import json
import sys
from pathlib import Path
path = Path(sys.argv[1])
if not path.exists():
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema_version": 1, "benchmark_kind": "paired_quality", "status": "failed",
                               "error": "Prediction preparation or evaluation failed; see the pair quality log."}) + "\n")
PY_QUALITY_FAILURE
                fi
            done
        done
    done
elif [[ "$mode" == qt-real ]]; then
    : "${BENCHMARK_DATA_ROOT:?Set BENCHMARK_DATA_ROOT to the directory containing mnist/ and blair/}"
    : "${BLAIR_CLASS_SPEC:?Set BLAIR_CLASS_SPEC to a reviewed Blair class specification}"
    for dataset in mnist blair; do
        extra=()
        if [[ "$dataset" == blair ]]; then
            extra=(--class-spec "$BLAIR_CLASS_SPEC" --hidden 64)
        fi
        for precision in float int8; do
            quantization=()
            if [[ "$precision" == int8 ]]; then quantization=(--quantized-training); fi
            run_profile "$dataset-$precision" --dataset "$dataset" --data-root "$BENCHMARK_DATA_ROOT/$dataset" \
                --epochs 5 --device cuda:0 --dtype float16 --cache CPU --cache-workers 0 --allow-nondeterministic \
                "${extra[@]}" "${quantization[@]}"
        done
    done
else
    : "${BENCHMARK_DATA_ROOT:?Set BENCHMARK_DATA_ROOT to the directory containing mnist/ and blair/}"
    : "${BLAIR_CLASS_SPEC:?Set BLAIR_CLASS_SPEC to a reviewed Blair class specification}"
    run_profile mnist-cpu --dataset mnist --data-root "$BENCHMARK_DATA_ROOT/mnist" --epochs 5
    run_profile mnist-cuda --dataset mnist --data-root "$BENCHMARK_DATA_ROOT/mnist" --epochs 5 --device cuda:0 --dtype float16 --cache CUDA --allow-nondeterministic
    run_profile blair-cuda --dataset blair --data-root "$BENCHMARK_DATA_ROOT/blair" --class-spec "$BLAIR_CLASS_SPEC" --epochs 5 --device cuda:0 --dtype float16 --cache CUDA --allow-nondeterministic
fi
"$benchmark_python" -m dev.benchmarks.summarize "$results" > "$results/summary.md"
cat "$results/summary.md"
exit "$status"
