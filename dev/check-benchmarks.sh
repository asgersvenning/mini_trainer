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
case "$mode" in cpu|gpu|real) ;; *) echo 'Mode must be cpu, gpu or real.' >&2; exit 2 ;; esac
mkdir -p -- "$results"
status=0
run_profile() {
    local profile="$1"
    shift
    if ! OMP_NUM_THREADS=1 "$benchmark_python" -m dev.benchmarks.run --output "$results/$profile" "$@" > "$results/$profile.log" 2>&1; then
        echo "Benchmark failed: $profile; see $results/$profile.log" >&2
        status=1
    fi
}
if [[ "$mode" == cpu ]]; then
    run_profile synthetic-cpu --device cpu
elif [[ "$mode" == gpu ]]; then
    for precision in float32 float16 bfloat16; do
        run_profile "synthetic-cuda-$precision" --device cuda:0 --dtype "$precision" --cache CUDA
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
