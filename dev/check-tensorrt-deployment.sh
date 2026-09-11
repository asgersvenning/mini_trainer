#!/usr/bin/env bash
# Build engines on the target and evaluate them using explicitly prepared environments.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
results="${1:?Supply a new results directory}"
if [[ -e "$results" || -L "$results" ]]; then
    echo 'Results directory must be new.' >&2
    exit 2
fi
mkdir -p -- "$results"
phase=configuration
trap 'code=$?; printf "{\"phase\":\"%s\",\"exit_code\":%d}\n" "$phase" "$code" > "$results/status.json"' EXIT
: "${BENCHMARK_PYTHON:?Set BENCHMARK_PYTHON to an explicitly prepared TensorRT/CUDA Python}"
: "${BENCHMARK_METRICS_PYTHON:?Set BENCHMARK_METRICS_PYTHON to an environment with mini_metrics}"
: "${TRT_BASELINE_MODEL:?Set TRT_BASELINE_MODEL to the floating ONNX model}"
: "${TRT_CANDIDATE_MODEL:?Set TRT_CANDIDATE_MODEL to the calibrated candidate ONNX model}"
: "${TRT_INFERENCE_MANIFEST:?Set TRT_INFERENCE_MANIFEST to the held-out inference manifest}"
: "${TRT_INPUTS:?Set TRT_INPUTS to representative named preprocessed NPZ inputs}"
: "${TRT_PROFILES:?Set TRT_PROFILES to the shared TensorRT profile JSON}"
export OMP_NUM_THREADS="${BENCHMARK_THREADS:-1}"
export PYTHONHASHSEED=0
phase=preflight
git rev-parse HEAD > "$results/revision.txt"
sha256sum -- dev/check-tensorrt-deployment.sh > "$results/harness.sha256"
"$BENCHMARK_PYTHON" -c 'import onnx, numpy, tensorrt, torch; assert torch.cuda.is_available(), "CUDA must be available"' > "$results/preflight.log" 2>&1
"$BENCHMARK_METRICS_PYTHON" -c 'from mini_metrics.data import MetricDF; from mini_metrics.metrics import MacroF1, evaluate_file' >> "$results/preflight.log" 2>&1
for role in baseline candidate; do
    phase="build-$role"
    model="$TRT_BASELINE_MODEL"
    if [[ "$role" == candidate ]]; then model="$TRT_CANDIDATE_MODEL"; fi
    "$BENCHMARK_PYTHON" -m dev.benchmarks.inference.tensorrt_build \
        --model "$model" --inputs "$TRT_INPUTS" --profiles "$TRT_PROFILES" \
        --output "$results/$role" --fp16 --device "${BENCHMARK_DEVICE:-0}" \
        --optimization "${TRT_OPTIMIZATION:-1}" --workspace-mib "${TRT_WORKSPACE_MIB:-1024}" \
        > "$results/build-$role.log" 2>&1
done
phase=evaluation
"$BENCHMARK_PYTHON" -m dev.benchmarks.inference.tensorrt_deployment \
    --baseline-build "$results/baseline" --candidate-build "$results/candidate" \
    --manifest "$TRT_INFERENCE_MANIFEST" --inputs "$TRT_INPUTS" --output "$results/evaluation" \
    --metrics-python "$BENCHMARK_METRICS_PYTHON" --threads "$OMP_NUM_THREADS" \
    --device "${BENCHMARK_DEVICE:-0}" > "$results/evaluation.log" 2>&1
phase=complete
