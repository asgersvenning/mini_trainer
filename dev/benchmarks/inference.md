# Inference benchmark workflow

All commands run from the checkout in explicitly prepared environments; they do
not install dependencies. Use fresh output directories and retain failed reports.
Use `python -m dev.benchmarks.inference.MODULE --help` for all options.

## Maintained image input preparation

Prepare calibration from training data and evaluation from the complete held-out
split, using the benchmark dataset manifest and exported model's preprocessing:

```bash
.venv/bin/python -m dev.benchmarks.inference.prepare_inputs \
  --dataset-manifest benchmark/dataset_manifest.json --data-root examples/blair \
  --export-manifest exported/manifest.json --output prepared-train \
  --split train --count 128 --seed 42 --batch-size 8 --score-semantics logits
.venv/bin/python -m dev.benchmarks.inference.prepare_inputs \
  --dataset-manifest benchmark/dataset_manifest.json --data-root examples/blair \
  --export-manifest exported/manifest.json --output prepared-val \
  --split val --batch-size 8 --score-semantics logits
```

Class order, source image hashes, input shape/dtype and selected identities are
checked and retained. Declare actual score semantics; arbitrary model outputs
are not necessarily logits. Omit `--count` for the complete split in manifest order.
Custom preprocessing needs an importable factory; the default uses repository
preprocessing on CPU with zero workers. Preparation checks declared split hashes
but cannot establish that an externally supplied split policy is appropriate.

## Maintained ONNX calibration command

```bash
.venv/bin/python -m dev.benchmarks.inference.onnx_calibration \
  --model exported/model.onnx --manifest prepared-train/manifest.json \
  --output qdq-cpu --threads 1
.venv/bin/python -m dev.benchmarks.inference.onnx_calibration \
  --model exported/model.onnx --manifest prepared-train/manifest.json \
  --output qdq-trt --threads 1 --activation-type int8 --symmetric-activations --float-bias
```

The CPU recipe defaults to unsigned activations, symmetric per-channel INT8
weights, quantized biases and Percentile 99.9 calibration over Conv/Gemm/MatMul.
The TensorRT candidate uses signed symmetric activation quantizers and floating
biases. Histogram symmetry is separate; the tested recipe does not select
`--symmetric-calibration`. Neither recipe is automatically qualified for a target.

Retain ordered NPZ batches/sample IDs, preprocessing, calibration provenance,
source/external-weight hashes, ranges and recipe report. Calibration inputs must
be finite and match all named model inputs. A calibration smoke run is not
held-out quality or proof of integer execution.

Native QT checkpoints can be explicitly materialized to floating deployment
weights before static calibration; see [export](../../docs/onnx.md). That route
changes the dynamic activation-quantization contract and is not a resume conversion.

## Composed CPU deployment comparison

```bash
.venv/bin/python -m dev.benchmarks.inference.cpu_deployment \
  --baseline exported/model.onnx --candidate qdq-cpu/model.onnx \
  --manifest prepared-val/manifest.json --inputs batch-one.npz \
  --output cpu-comparison --threads 1 --trials 3 --warmup 3 --repeats 31 \
  --require-provider-op QLinearConv --require-provider-op QGemm \
  --metrics-python /path/to/metrics-env/bin/python
```

The manifest supplies held-out identities, class ordering and score semantics;
the separate NPZ supplies the representative resource-measurement shape. Use the
same preprocessed arrays for both models and retain their selection provenance.
Operator requirements are explicit examples; select the actual required operators
for your candidate rather than assuming every architecture uses QGemm.

Quality, placement and each resource trial run separately. Trial order alternates,
and hashes/settings/runtime identities must agree. CPU ratios compare separate-
process warm latency medians, process RSS and approximate peak RSS. Placement
counts show remaining floating operations; requiring one integer operator does
not prove complete quantization. Requested trial/warmup/repeat budgets are retained.

## Composed TensorRT deployment evaluation

Prefer [the target harness](reporting.md#opt-in-target-gpu-workflow), which rebuilds
both engines on the target and evaluates them. For separately prepared engines:

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python -m dev.benchmarks.inference.tensorrt_deployment \
  --baseline-build results/fp16 --candidate-build results/int8 \
  --manifest prepared-val/manifest.json --inputs prepared-val/batch-00000.npz \
  --output trt-comparison --metrics-python /path/to/metrics-env/bin/python \
  --trials 3 --warmup 10 --repeats 31 --memory-runs 20 --threads 1
```

Build directories must contain matching `model.engine`, `report.json` and
`layers.json`, created by `inference.tensorrt_build`. Use matching profiles/settings
and a practical FP16 baseline. Engines are target-specific; keep ONNX external
weights beside source models. All resource shapes must fit both profiles.

Evaluation checks build/engine/input identities, then collects held-out quality,
adjacent paired GPU latency and isolated single-engine memory trials. Both engines
coexist during timing. Memory snapshots come from separate processes and are
not transient peaks or per-process device allocation.

## Individual tools

| Module in `dev.benchmarks.inference` | Purpose |
| --- | --- |
| `onnx_inference` | Warm timing, output parity and provider/operator placement |
| `onnx_cpu_memory` | Fresh-process CPU latency, RSS and approximate peak |
| `tensorrt_build` | Build engine from ONNX with explicit profiles; retain layer inspection |
| `tensorrt_pair` | Adjacent paired engine timing |
| `tensorrt_memory` | Isolated engine memory stages |
| `dataset_inference` | Full held-out predictions and canonical CSVs |
| `inference_pair` | Compose baseline/candidate collection and quality |
| `quality_compare` | Five-metric comparison through prepared `mini_metrics` |

## Interpretation and retention

Read Macro-F1, Macro-Recall, Macro-Precision, Coverage and Theil's U at every
classification level. Preserve undefined metrics and inspect quality together with
resource benefit. `evaluated` means execution completed, not production acceptance.

Retain source bundles, manifests/NPZs and complete output directories: child hashes,
commands/logs, predictions, metrics, raw timings, failures and `summary.md`.
[Reporting](reporting.md) projects compact visible history without replacing these
reproduction inputs. Host timings exclude preprocessing/loading; qualify that cost
separately on idle target hardware. CPU tests and x86 measurements do not establish
ARM execution; provider fallback is not integer GPU inference.
