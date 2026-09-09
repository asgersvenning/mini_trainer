# Training and loading benchmarks

Use prepared environments and fresh result directories. Start with the shared
CPU oracle in [the index](README.md), then use the representative profile below.
[Findings](../../docs/benchmarks.md) and [remaining work](../../docs/quantization-roadmap.md)
are maintained separately from these commands.

## Representative paired EfficientNetV2 training profile

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=0 \
BENCHMARK_DATA_ROOT=/path/to/examples \
BLAIR_CLASS_SPEC=/path/to/reviewed/class_spec.json \
BENCHMARK_METRICS_PYTHON=/path/to/metrics-env/bin/python \
bash dev/check-benchmarks.sh qt-efficientnet fresh-results
```

This runs pretrained EfficientNetV2-S with symmetric normalized flat/hierarchical
heads, full/frozen backbones, seeds 42/43/44 and floating BF16/native INT8:
24 sequential training processes followed by twelve held-out comparisons.
Defaults: five epochs, batch 32, 128px images, MuonAuxAdamW, learning rate 0.01,
CPU caching, zero workers and no compilation. Odd seeds reverse pair ordering.
Pretrained initialization can download weights; prepare its cache for offline use.

| Override | Default | Selection |
| --- | --- | --- |
| `BENCHMARK_HEAD` | `both` | `flat`, `hierarchical`, `both` |
| `BENCHMARK_TRAINING_MODE` | `both` | `full`, `frozen`, `both` |
| `BENCHMARK_SEEDS` | `42 43 44` | Unique nonnegative integers |
| `BENCHMARK_EPOCHS` | `5` | Positive fixed schedule budget |
| `BENCHMARK_PYTHON` | `.venv/bin/python` | Prepared training interpreter |

Choose budgets before inspecting held-out results. The profile evaluates `last.pt`,
retaining reports, checkpoints, epoch CSVs, predictions, manifests and failures.
`summary.md` reports Macro-F1, Macro-Recall, Macro-Precision, Coverage and Theil's U
at every level; deltas are multiplied by 100, including Theil's U. Undefined values
remain explicit. Failed pairs return nonzero without hiding the other pairs.

## Individual configurations

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 .venv/bin/python -m dev.benchmarks.training.run \
  --dataset blair --data-root examples/blair \
  --class-spec examples/blair/blair_model/class_spec.json \
  --backbone efficientnet_v2_s --head hierarchical --hidden symmetric --normalized \
  --pretrained --image-size 128 --epochs 5 --batch-size 32 --seed 42 \
  --device cuda:0 --dtype bfloat16 --cache CPU --cache-workers 0 --num-workers 0 \
  --quantized-training --output fresh-int8-run
```

Omit `--quantized-training` for its floating pair. Match every other setting and
repeat seeds with alternating order. `--fine-tune` freezes backbone parameters
but retains normal training-mode BatchNorm/dropout; it is different from the
large-head capacity probe's evaluated frozen backbone. Compilation must be selected
identically for both precisions. `--help` lists the supported compilation modes.

CPU float32 without QT verifies pipeline behavior; native INT8 execution needs
CUDA. Small random backbones verify execution, not useful convergence. Flat and
hierarchical comparisons retain the same leaf indices, reviewed taxonomy and splits.

## Capacity and bottleneck probes

Run each probe with `python -m dev.benchmarks.<package>.<module> --help` for its
arguments. These are focused diagnostics, not substitutes for paired training.

| Module | Measures |
| --- | --- |
| `training.large_head_training` | EfficientNetV2 large-head setup, steps and memory; `--frozen` evaluates the backbone |
| `training.quantized_training` | Integer linear forward/backward and updates against floating compute |
| `training.quantization` | Separate CPU PTQ/QAT behavior |
| `data.loader` | Cached loader throughput |
| `data.cache` | Cache construction and host/shared-memory costs |
| `data.reader` | Streaming image decode/resize |
| `data.transfer` | Host/device transfer and overlap |

```bash
OMP_NUM_THREADS=1 .venv/bin/python -m dev.benchmarks.data.loader
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
  .venv/bin/python -m dev.benchmarks.training.quantized_training --width 2048 --batch-size 512
```

## Training predictions to paired quality evaluation

`training.training_predictions` converts existing runner predictions and dataset
manifests into the canonical paired input contract; `inference.quality_compare`
executes the five metrics in an explicitly selected `mini_metrics` environment.
The shared `qt-efficientnet` profile composes both. Use their `--help` for manual
re-evaluation. Match sample identities, class order, split and score semantics;
never substitute a classification-accuracy delta for the full metric comparison.

## Measurement contract

Use an otherwise idle allocation. Record hardware, versions, model/input hashes,
threads, workers, precision coverage and seeds. Separate cold setup/first step,
warm training, loading/transfers, allocated/reserved peaks and whole-call time.
Graph replay can trade setup or reserved memory for warm speed. Native QT retains
floating convolutions, gradients and optimizer states; report actual coverage.

Epoch statistics from old runs with all statistic loggers disabled are invalid
convergence evidence, although independently evaluated final weights remain usable.
Historical optimizer/RNG state retention is limited; see [artifacts](../../docs/quantization-artifacts.md).
A few quality points are tolerable only with a substantial measured cost/speed or
memory benefit. Target hardware qualification remains open.
