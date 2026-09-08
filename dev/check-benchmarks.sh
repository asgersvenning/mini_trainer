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
case "$mode" in cpu|gpu|real|qt|qt-real|qt-dense|qt-large-batch) ;; *) echo 'Mode must be cpu, gpu, real, qt, qt-real, qt-dense or qt-large-batch.' >&2; exit 2 ;; esac
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
elif [[ "$mode" == qt-large-batch ]]; then
    : "${BENCHMARK_DATA_ROOT:?Set BENCHMARK_DATA_ROOT to the directory containing mnist/}"
    for seed in 42 43 44; do
        precisions=(float int8)
        if (( seed % 2 )); then precisions=(int8 float); fi
        for precision in "${precisions[@]}"; do
            quantization=()
            if [[ "$precision" == int8 ]]; then quantization=(--quantized-training); fi
            run_profile "mnist-large-batch-$precision-seed$seed" --dataset mnist --data-root "$BENCHMARK_DATA_ROOT/mnist" \
                --seed "$seed" --model-profile dense --optimizer sgd --learning-rate 0.3 --epochs 60 --batch-size 512 \
                --compile --compile-optimizer --device cuda:0 --dtype float16 --cache CPU --cache-workers 0 \
                --allow-nondeterministic "${quantization[@]}"
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
