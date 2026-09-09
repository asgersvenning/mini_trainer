# Reproducible benchmark progression

Start with a synthetic oracle, then MNIST, then hierarchical Blair. CPU is the fast
continuous check; GPU profiles validate additional behavior. Birds and iNaturalist
2021 remain optional larger follow-ups. Notebook settings and timing claims are
references, not acceptance criteria.

See [visible results and coverage](../../docs/benchmarks.md) for initial observations,
known issues, reporting and storage limits.

## Shared pipeline commands

Use the existing environment; these commands never install or synchronize packages:

```bash
bash dev/check-benchmarks.sh cpu /tmp/benchmarks-cpu
bash dev/check-benchmarks.sh gpu /tmp/benchmarks-gpu
BENCHMARK_DATA_ROOT="$PWD/examples" \
BLAIR_CLASS_SPEC="$PWD/examples/blair/blair_model/class_spec.json" \
    bash dev/check-benchmarks.sh real /tmp/benchmarks-real
bash dev/check.sh test tests/test_benchmark_synthetic.py tests/test_benchmark_datasets.py
```

Each results directory must be new. Set `BENCHMARK_PYTHON` to use a different prepared
interpreter. CPU runs the synthetic float32 profile. GPU runs synthetic float32,
float16 AMP and bfloat16 AMP with CUDA caching. Real runs MNIST on CPU and CUDA,
then hierarchical Blair on CUDA. Profiles run sequentially, retain failures, and
return a nonzero exit code if any profile fails. No requested GPU silently falls
back to CPU. The real profile requires existing datasets and a saved Blair class
specification; it performs no downloads or online taxonomy queries.

Individual runs are configurable:

```bash
.venv/bin/python -m dev.benchmarks.run --output /tmp/oracle \
    --seed 42 --threads 1 --device cpu
.venv/bin/python -m dev.benchmarks.run --dataset mnist \
    --data-root examples/mnist --output /tmp/mnist-amp --epochs 5 \
    --device cuda:0 --dtype float16 --cache CUDA --allow-nondeterministic
```

`--num-workers` defaults to zero and is always explicit in reports. CUDA caching
forces zero effective loader workers. Threads default to one. Cache modes are
`NONE`, `RAM` and `CUDA`. The CPU float32 profile and synthetic GPU profiles use
strict deterministic algorithms. Real GPU profiles explicitly allow nondeterministic
operations because CUDA adaptive-pooling backward lacks a deterministic implementation.
AMP, CUDA caching and algorithm determinism are separate report fields.

## Dataset and training contracts

The synthetic generator uses four balanced classes encoding two binary factors in
red and green channels of 8-by-8 RGB images. Intensities are 32 or 224 with bounded
noise of at most 16. Thresholding channel means at 128 is an exact 100% oracle;
uniform guessing achieves 25% in expectation. Blue contains nuisance noise. A parent
label encodes the red factor, ready for a future synthetic hierarchical profile.

Independent streams keyed by seed, split, class and sample generate 128 training,
32 validation and 64 test images. The synthetic runner uses flat labels, a channel-mean
backbone and an unnormalized linear classifier. Fixed 12-epoch runs train with
MuonAuxAdamW at learning rate 0.1. The exact oracle quality gate fails if held-out
accuracy is below 100%.

MNIST and Blair use a tiny randomly initialized convolutional backbone, five epochs
and learning rate 0.01 in the shared profile. MNIST uses 28-pixel inputs and a flat
head; Blair uses 64-pixel inputs and the normalized hierarchical head. Blair's
reviewed specification must exactly cover the training class names. It is included
in the manifest so mappings cannot change silently across runs.

For real datasets, 20% of each class's unique training files become validation data,
using seeded shuffling. Byte-identical duplicates stay together. Training files
identical to supplied test files are excluded from the index and recorded; the
source files and official test split remain untouched. Conflicting cross-split
content fails validation. Hashes cover encoded file bytes, not perceptual duplicates.

Every profile uses the actual mini_trainer training, checkpoint reconstruction and
inference loader. The final checkpoint is evaluated after a fixed epoch budget;
held-out labels are never used for checkpoint or hyperparameter selection. There
is no augmentation, EMA or regularization in this baseline. Real-data completion
has no quality acceptance threshold yet. Declare such thresholds and comparisons
before attempting improvements; do not tune against these test scores.

## Reproduction artifacts

Each run writes `report.json`, `predictions.npz` with sample paths and per-level
scores/targets, dataset/split manifests, and a normal training directory containing
resolved configuration and checkpoints. Synthetic images can be regenerated;
real images remain external. Scores retain the model's evaluation-forward semantics.
The report contains source, lockfile, dataset and checkpoint hashes, versions,
configuration, hardware, coverage flags, quality and synchronized training-call wall
time. Peak CUDA allocated memory covers training, not total device/process memory.

Wall time includes initialization, training, validation, logging and checkpoint
writes. It excludes interpreter startup, inventory/generation and final inference.
It is not pure throughput or loader-wait time. Dedicated profiling and repeated,
matched experiments are still needed to establish speed improvements.

Tests verify exact repeated CPU predictions within one environment, matching dataset
hashes, independent splits, oracle correctness, explicit GPU failure and taxonomy
contracts. They do not guarantee byte-identical checkpoints or cross-version/GPU
numerical identity. Reports explicitly mark unexercised features; do not infer full
module coverage from a passing profile.

## Continuous execution and visible reporting

`.github/workflows/benchmarks.yml` runs the CPU oracle on pull requests, master pushes
and weekly. It uses a locked CPU environment and uploads results even on failure.
The Actions run page contains a readable summary and artifacts with 90-day retention.
The README links to that history. The source checkout is not modified to publish results.
These are initial reporting facilities; durable archival and a historical dashboard
remain follow-up work.

GPU execution uses a configured self-hosted runner labeled `self-hosted`, `linux`,
`gpu`. It creates a disposable uv environment with an explicitly selected CUDA extra,
leaving the runner's existing environments alone. Enable it manually using the `gpu`
workflow input, or set repository variable `ENABLE_GPU_BENCHMARKS=true` for weekly
execution. Set `BENCHMARK_CUDA_BACKEND` for scheduled runs (default `cu130`).

For real-data jobs, enable input `real_data` together with `gpu`, or set
`BENCHMARK_REAL_DATA=true`. Set repository variables `BENCHMARK_DATA_ROOT` and
`BLAIR_CLASS_SPEC` to stable dataset and specification paths on that runner, outside
the checkout. The workflow does not assume these resources already exist and cannot
validate GPU behavior on a CPU-only hosted runner. No GPU job has been dispatched
from this development session; the equivalent local commands have been exercised.

References: [Actions job summaries](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-commands#adding-a-job-summary)
and [artifact retention](https://github.com/actions/upload-artifact#retention-period).

## Quantized training and loader performance

Actual quantized training has an initial Linear integration. PTQ/QAT and AMP do
not establish reduced training memory or faster training. Broader coverage and
real-workload speedups remain implementation targets.

Two developer probes make the remaining work measurable:

```bash
OMP_NUM_THREADS=1 .venv/bin/python -m dev.benchmarks.loader
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1 \
    .venv/bin/python -m dev.benchmarks.quantized_training
# Also test smaller GEMMs and full precision; benefit is workload-dependent:
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m dev.benchmarks.quantized_training --width 2048 --batch-size 512
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m dev.benchmarks.quantized_training --dtype float32
```

The QT probe stores weights and saved linear inputs in INT8, uses scaled INT8
forward/input-gradient/weight-gradient GEMMs, and writes SGD updates back using
stochastic rounding. There is no retained floating-point master weight copy.
Gradients and update arithmetic remain floating point. It uses TorchAO's
experimental weight storage and Triton kernels with `torch.compile` fusion.
The experimental parameter dispatch supports SGD (including momentum/Nesterov
and weight decay) and AdamW. CPU tests check update error against floating-point
optimizer math and exact next-step continuation when weights, optimizer state
and RNG are restored together. Foreach parameter updates currently dispatch per
tensor; no fused-optimizer speed benefit is claimed. This does not establish the
repository's complete optimizer or checkpoint contracts. The backend now has an
initial [model and trainer integration](../../docs/quantized-training.md), including
checkpoint restoration. The linear-stack MSE remains a kernel probe, not a
convergence study.

Local RTX 3080 Ti evidence (four 4096-wide layers, batch 2048, FP16 input/output,
three warm-up steps, ten measured forward/backward/SGD steps): FP16 took 29.10 ms
per step and peaked at 386,139,648 allocated bytes; INT8 took 12.92 ms and peaked
at 302,302,720 bytes. Stored weight bytes fell from 134,217,728 to 67,141,632.
These historical measurements predate the backward-scale underflow correction
and must not be used as evidence for the corrected implementation. They are
retained to document the investigation, not as valid training speedup claims.
The smaller 2048-wide/batch-512 probe was slower in INT8, and the unfused prototype
used more peak memory. Preserve those negative results when choosing dispatch.

The loader probe compares identical uint8 cached batches against the previous
scalar-fetch/default-collate route. Batched `index_select` avoids restacking and
measured about 1.5x faster locally at 2048 RGB 64x64 images, batch 64, one thread,
zero workers. It excludes cache construction, preprocessing, GPU copies and model
compute; no end-to-end training gain is implied.

Loaders retain their sampling/drop-last policies and worker caps. CPU batches
bound for CUDA are pinned after gathering, including CPU-cached and inference
batches. Optional `prefetch_factor` and `multiprocessing_context="spawn"` pass
through loader builders; both are inactive with zero workers. Spawn is useful
when parent code already has background threads, because fork may deadlock.
Use importable/pickleable readers and hooks with spawn.

For the separate PTQ/QAT baseline comparison:

```bash
.venv/bin/python -m dev.benchmarks.quantization --baseline /path/to/synthetic-cpu --output /tmp/int8-synthetic
.venv/bin/python -m dev.benchmarks.quantization --baseline /path/to/mnist-cpu --data-root examples/mnist --output /tmp/int8-mnist
```

This checks baseline checkpoint/manifest/file hashes, selects training-only
calibration samples, performs two small QAT epochs, then evaluates reloaded native
integer artifacts on the held-out split. The report and per-level predictions
retain baseline/PTQ/QAT results. It does not establish QT training speed or memory.

The QT kernel probe also accepts `--optimizer-name adamw --weight-decay 0.1`
or `--momentum 0.9 --weight-decay 0.1` for SGD. AdamW uses an explicit
`--epsilon 1e-4` for both compared models: its usual `1e-8` underflows in FP16
state. Use `--dtype float32 --epsilon 1e-8` to test ordinary float32 optimizer
state. This changes the experimental recipe, not the training CLI defaults.
CUDA regression tests exercise ordinary `nn.Linear` dispatch with non-square
weights, bias, batched inputs and masked classifier rows. Model tests also cover
MuonAuxAdamW and controlled `mt_train` resume. Row-wise weight normalization is now covered separately. Convolutional
QT, DDP and arbitrary stochastic continuation remain unverified.

The original FP16 backward scale products could underflow before quantization,
suppressing gradients from mean-reduced losses. The corrected kernel forms
those products and their row scales in float32 while retaining INT8 GEMMs and
saved inputs. A CUDA regression compares input/weight gradients at both ordinary
and `1e-6` upstream gradient magnitudes. This was discovered through the AdamW
learning probe, beyond the original large-gradient numerical test.

The apparent compiled zero-gradient failure was traced to an AOT disk-cache hit
for the old FP16 backward scale products, not the corrected source. The
experimental tensor now supplies a stable key containing its implementation
source digest and tensor metadata. Changing backward/dispatch code invalidates
that key; changing weight values does not. CUDA tests compare compiled and eager
parameter gradients for a small mean-reduced loss. The probe retains its
missing/zero-gradient gate, and records warm-up loss trajectories before timing.

With the versioned key, the compiled two-layer 2048-wide, batch-512 AdamW run
(decay 0.1, epsilon 1e-4, three warm-up and three measured steps) ended at MSE
0.36759 for INT8 versus 0.36605 for FP16. INT8 took 2.16 ms/step and peaked at
103,033,344 bytes versus 1.53 ms and 92,539,392 bytes. Compilation improves on
the eager INT8 result below, but this remains a negative speed and memory result
against compiled FP16. Larger-workload and end-to-end gains require measurement.

After the scale correction, the eager two-layer 2048-wide, batch-512 AdamW probe
(seed 42, three warm-up and three measured steps, decay 0.1, epsilon 1e-4)
reduced MSE from 0.99986 to 0.25309 in INT8, versus 0.99959 to 0.25218 in
FP16. INT8 took 5.85 ms/step and peaked at 171,218,944 bytes, versus 2.39 ms
and 111,412,224 bytes for FP16. This establishes similar short-run learning in
that diagnostic, but is a negative speed and peak-memory result. Stored weight
bytes alone fell from 16,777,216 to 8,396,800; that does not complete the QT goal.

Repeating the original large compiled SGD probe after both corrections (four
4096-wide layers, batch 2048, three warm-up and ten measured steps) passed the
nonzero-gradient gate. INT8 took 18.12 ms/step and peaked at 319,063,552 bytes,
versus 30.39 ms and 386,139,648 bytes for FP16: about 1.68x faster and 17% less
peak allocated memory on this RTX 3080 Ti workload. Weight storage remained
67,141,632 versus 134,217,728 bytes. Final MSE was 0.99880 versus 0.99870; the
small SGD updates in this probe do not establish convergence. This replaces the
pre-correction large-workload timing above. Real-model training, optimizer-state
memory, data loading and task quality still require end-to-end validation.

## Bounded cache construction

CPU/CUDA cache construction now reserves four available CPUs and caps automatic
reader threads at 16 (previously up to 128, reserving two). Four or fewer CPUs
select synchronous construction. `LazyDataset(..., cache_workers=0)` and
`get_dataset_dataloader(..., cache_workers=0)` explicitly disable cache reader
threads; positive values override the automatic selection. Existing builder
keyword forwarding supports this option. DataLoader worker selection is separate.

At most twice the selected reader count is submitted ahead, plus a write batch
of at most 64 samples. There is no unbounded reorder/write queue or daemon writer.
Readers run once per sample, including the first shape probe. Read errors,
inconsistent sample shapes/structures and write errors propagate to the caller;
pending work is cancelled and active readers finish before construction returns.
Cached ordering, labels and dtypes remain unchanged for valid inputs.

Compare against the previous implementation using the developer benchmark:

```bash
git show 7a095a2:mini_trainer/data/io.py > /tmp/cache-baseline.py
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' .venv/bin/python -m dev.benchmarks.cache --baseline-source /tmp/cache-baseline.py
# Explicit synchronous cache construction:
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' .venv/bin/python -m dev.benchmarks.cache --workers 0
```

`--baseline-source` executes developer-supplied Python source. The benchmark uses
seed 42, 2048 samples cycling through 64 generated RGB 128x128 PNGs, a warm
filesystem cache, one PyTorch thread and five measured repeats in alternating
order. File generation is excluded. It verifies image and label output and
records source hashes. This is cache-construction timing, not training throughput.

On the local 20-CPU allocation, automatic construction used 16 reader threads
versus the previous 18. The final comparison measured 1,722 versus 1,686 samples
per second (about 1.02x); an earlier repeat was effectively tied. Treat this as
similar throughput, not evidence of a meaningful speedup. The concrete gain is
bounded read-ahead and reliable failure handling, with explicit synchronous
construction available for shared-node environments. CUDA checks also verify
pinned CPU transfer batches and exact CPU/CUDA cache contents.

## Integrated QT dataset profiles

### Representative EfficientNetV2 configuration

The runner accepts a registered `--backbone` independently of the dataset and
`--head flat|hierarchical`. `--hidden symmetric` uses the backbone embedding width;
`--normalized` enables the normalized head for flat classification too. Existing
benchmark defaults are preserved. Explicit backbones start without pretrained
weights unless `--pretrained` is supplied; that flag permits a download. Reload
uses the saved checkpoint without requesting pretrained initialization again.

For a matched Blair comparison on an available CUDA machine:

```bash
for head in flat hierarchical; do
  for precision in float int8; do
    qt_args=()
    if [[ "$precision" == int8 ]]; then qt_args=(--quantized-training); fi
    CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1 \
      .venv/bin/python -m dev.benchmarks.run \
      --dataset blair --data-root examples/blair \
      --class-spec examples/blair/blair_model/class_spec.json \
      --backbone efficientnet_v2_s --head "$head" --hidden symmetric --normalized \
      --image-size 128 --epochs 5 --batch-size 32 --seed 42 \
      --device cuda:0 --dtype float16 --cache CPU --cache-workers 0 --num-workers 0 \
      "${qt_args[@]}" --output "tmp-efficientnet/$head-$precision"
  done
done
```

Each output directory must be new. This is an eager baseline; add identical
compilation settings to both precisions when measuring compilation. Repeat paired
seeds with alternating precision order before interpreting quality or speed.
Both heads use identical leaf indices and the same dataset manifest, including
the reviewed parent taxonomy. Flat training projects the taxonomy to the leaf
mapping; it does not regenerate splits or reorder classes. Reports retain the
backbone, effective hidden width, normalization, image size and initialization
choice. Held-out inference currently checks scores and quality; it is not an
ONNX latency benchmark.

For offline pipeline verification use `--device cpu --dtype float32` without
`--quantized-training`. Smaller image sizes or short runs may check execution,
but must be reported as diagnostic settings rather than the production workload.
The CPU integration test exercises both real EfficientNetV2 heads with synthetic
image files and a deliberately reordered taxonomy; it checks training, reload,
predictions, class order and identical split manifests without a download.

### Validation when target hardware is unavailable

Use the local GPU to vary batch size, resolution, cache mode and worker count
one at a time. Compare each quantized run with the same floating configuration;
record out-of-memory failures, conversion costs and floating operator coverage.
Smaller memory budgets and constrained CPU affinity can exercise resource limits,
but cannot reproduce a different GPU architecture, bandwidth or ARM instruction
set. Keep compiler warmup separate from steady-state measurements. The CPU suite
is a correctness check, not an estimate of Raspberry Pi latency.

When machines become available, use the same revision, lock file, dataset
manifest, checkpoints and commands, with a separately selected compatible backend:

- HPC: run paired PyTorch training on the allocated GPU/CPU resources, recording
  exact hardware, CPU affinity/quota, software versions and whether storage is
  local or shared. Measure each GPU generation separately. Multi-GPU QT remains
  unsupported and requires its own implementation and validation.
- Local batch processing: repeat training and frozen-backbone fine-tuning as
  distinct workloads. Then export and measure ONNX GPU inference with provider
  placement evidence; a listed provider alone does not prove GPU execution.
- Edge: transfer the export, preprocessing recipe, class mapping and fixed input
  samples to the ARM device. Verify scores before measuring batch-one latency,
  sustained throughput and process memory under explicit thread counts.

Floating and native INT8 EfficientNetV2 checkpoints now have locally verified
[ONNX export paths](../../docs/onnx.md). Native export requires an explicit CUDA
reference; it retains floating convolutions and integer head products. Deployment
quantization and runtime placement must still be validated for each target
provider; these training commands do not establish those results.
Retain reports, logs, failures and predictions with the existing benchmark
summary/artifact workflow so external runs can be reviewed without machine access.

Install the optional `quantization` extra with the intended CUDA backend explicitly
selected (see the repository README), then use new output directories:

```bash
CUDA_VISIBLE_DEVICES=0 TORCHINDUCTOR_COMPILE_THREADS=1 \
    bash dev/check-benchmarks.sh qt /tmp/benchmarks-qt
CUDA_VISIBLE_DEVICES=0 TORCHINDUCTOR_COMPILE_THREADS=1 \
    BENCHMARK_DATA_ROOT=/path/to/datasets BLAIR_CLASS_SPEC=/path/to/class_spec.json \
    bash dev/check-benchmarks.sh qt-real /tmp/benchmarks-qt-real
```

The first command pairs floating and INT8 synthetic training. The second pairs
MNIST and hierarchical Blair, using a reviewed existing Blair class specification.
Both use FP16 AMP, CPU caching and zero cache/loader workers; Blair uses hidden
size 64 in both paths to exercise both a hidden Linear and its normalized head. Earlier recorded
profiles quantized only the hidden layer; inspect each report for actual coverage. The runner exposes `--hidden`, `--batch-size`, `--compile` and
`--cache-workers` for explicit additional profiles. Defaults remain unchanged.
`--cache RAM` is retained as an alias for `CPU`.

For Actions, enable `gpu=true` and `qt=true`, adding `real_data=true` for both
real datasets. Scheduled GPU runs can enable `BENCHMARK_QT=true` together with the
existing GPU and real-data variables described above. The disposable GPU
environment installs the quantization extra only when QT is enabled. CUDA model
regressions precede the paired profiles; summaries and artifacts retain measured
coverage, storage, timing and failures. This workflow wiring has been checked
locally but has not been dispatched on a self-hosted runner from this session.

[Local measured results](../../docs/benchmarks.md#integrated-int8-training) include
slower QT training on these small workloads. Whole-model compilation defaults off;
first-use kernel compilation is still included in wall time. Compare matching
configurations and compiler cache conditions before making performance claims.

## CUDA transfer overlap

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 .venv/bin/python -m dev.benchmarks.transfer
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 .venv/bin/python -m dev.benchmarks.transfer --backward
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 .venv/bin/python -m dev.benchmarks.transfer --dtype float32
```

The probe compares the ordinary loader with `cuda_prefetch=True`, using the same
CPU-cached uint8 images, pinned batches and fixed ResNet18 weights. BatchNorm is
frozen; the optional backward pass includes gradient computation but no optimizer
updates. Model/cache construction is excluded, one trial warms both paths, and
five measured trials alternate order. Every run checks bitwise-identical outputs.
JSON records timings, hardware, PyTorch version and peak allocated CUDA memory.
This measures transfer plus model computation, not convergence or complete trainer
throughput. The dataset harness also accepts `--cuda-prefetch` for actual training,
checkpoint reload and held-out inference profiles, including INT8 models.

On the RTX 3080 Ti Laptop GPU with PyTorch 2.12.0+cu130, one CPU thread, 512 images
at 224x224, batch size 32 and FP16 AMP, median throughput increased from 3,206 to
3,310 images/s for inference (1.03x), and 1,061 to 1,078 images/s for
forward/backward (1.02x). Peak CUDA allocation increased by 9,569,792 bytes in each
case, consistent with staging an additional input batch plus small overheads.
These modest local observations are diagnostic, not portable performance gates.

The matching float32 inference probe was slower with lookahead: 2,002 versus
1,918 images/s (0.96x). Keep the default off unless measurements on the intended
workload justify the additional stream and memory. The integrated INT8 synthetic
training/checkpoint/inference profile passed its 100% oracle gate with prefetch
and reproduced the earlier QT held-out scores bit for bit; that establishes
compatibility, not a training speedup.

### Streaming image-reader comparison

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 .venv/bin/python -m dev.benchmarks.reader \
    --data-root examples/blair/test --samples 128 --size 224 --batch-size 16 \
    --workers 1 --repeats 7 > /tmp/blair-reader.json
```

Repeat with `--workers 0` or `--data-root examples/mnist/test`. This uses the
actual streaming inference loader and compares the former torchvision resize
path with the current reader. Every batch must match exactly before timing.
JSON includes relative file names, content hashes, dependency versions and every
timing sample. Paths are sorted and the first requested number of JPEG/PNG files
is used; files are never modified.

Timing includes file reads, decoding, nearest resize, assembly and worker IPC.
The equivalence pass warms the filesystem cache and persistent workers, so these
are not cold-disk or worker-startup measurements. H2D and model compute are excluded.
See the [reader findings](../../docs/benchmarks.md#uint8-nearest-resize-in-the-streaming-reader).

### Worker batch assembly

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 .venv/bin/python -m dev.benchmarks.loader \
    --workers 1 --cache cpu --samples 512 --size 224 --batch-size 32 --repeats 7
```

Use `--cache none` to probe uncached tensor assembly. These are synthetic tensor
readers, so neither mode measures image decoding. The probe verifies identical
batches, warms persistent spawn workers and then alternates scalar/batched passes.
Timing includes worker IPC but excludes startup, cache construction, preprocessing,
H2D and model compute. Worker count is explicit and defaults to zero; cached
loading does not automatically benefit from additional workers.

Repository CPU-cache gathers now write directly into shared storage inside
workers. External collators retain their own allocation path. Main-process
pinning and CUDA-cache behavior remain separate. See the [measured worker
results](../../docs/benchmarks.md#shared-storage-for-cached-worker-batches).

### Direct pinned gathering

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 .venv/bin/python -m dev.benchmarks.loader \
    --pin-batches --samples 512 --size 224 --batch-size 32 --repeats 7
```

This compares gathering into pageable memory followed by pinning against gathering
directly into pinned storage. Both return identical pinned batches; measurements
exclude cache construction, H2D and model compute. On the RTX 3080 Ti Laptop host,
seven alternating measured trials after warmup gave 30,983 versus 72,680 images/s
(2.35x). This is a loader-only result, not a claim of a 2.35x training speedup.
The shared CUDA-target CPU-cache loader enables direct pinned gathering when
`num_workers=0`; workers retain parent-side pinning.

The transfer probe now measures four variants in the same run: the former
same-stream path, transfer prefetch, direct pinned gathering, and their combination.
This separates the effects of eliminating a CPU copy and overlapping H2D with
compute. Each variant still checks exact output equality against the others.

In the four-way ResNet probe on the same host, float32 inference measured 2,050
images/s for the former path, 2,063 for direct pinned gathering, and 2,107 for
pinned gathering plus prefetch. FP16 forward/backward measured 1,040, 1,009 and
1,028 images/s respectively. These compute-inclusive differences are small and
include regressions despite the clear loader-only improvement. Eliminating the
CPU copy does not establish a training speedup for compute-bound models.

## Dense real-data QT comparison

```bash
CUDA_VISIBLE_DEVICES=0 TORCHINDUCTOR_COMPILE_THREADS=1 BENCHMARK_DATA_ROOT=/path/to/datasets \
    bash dev/check-benchmarks.sh qt-dense /tmp/benchmarks-qt-dense
```

This profile uses MNIST with a dense spatial image MLP: input pooling to 28x28,
three 2048-wide Linear/ReLU layers, and the repository's classifier head. It is
an explicit compute profile, not a proposed CNN replacement. Both paths use
15 epochs, batch size 128, seed 42, SGD with momentum 0.9, head LR 0.3 (backbone
LR 0.1), zero weight decay, FP16 AMP and a CPU cache with zero workers. The model
is compiled; optimizer updates are currently eager. No pretrained weights,
augmentation or test-set selection is used. The ordinary benchmark defaults
remain unchanged. `--model-profile`, `--optimizer` and `--learning-rate` also
allow explicit additional recipes whose configuration is retained in reports.

QT plus real-data Actions profiles now include this pair and retain its reports
and summaries. This wiring has not been dispatched from this development session.

The benchmark logger preserves CUDA peaks before every batch/phase reset and records
synchronized train/evaluation batch-loop timing and allocation peaks. The report
also retains total training-call wall time, including setup, figures, checkpoint
writes and first-use compilation. Per-phase timing excludes figures/checkpoints;
first-epoch compilation remains visible. Earlier dataset reports read the peak
only after the logger's final reset; their CUDA readings are now marked
`unverified`, and cannot establish whole-run memory reductions. This correction
does not affect the standalone kernel and transfer probes, which do not use that
logger. It also does not affect recorded accuracy or physical parameter storage.

### Multi-seed large-batch comparison

```bash
CUDA_VISIBLE_DEVICES=0 BENCHMARK_DATA_ROOT=/path/to/datasets \
    bash dev/check-benchmarks.sh qt-large-batch /tmp/benchmarks-qt-large-batch
```

This additional profile uses the same dense model, optimizer, learning rates and
data preparation with batch size 512, 60 epochs and both model and optimizer
compilation. It runs matched float/INT8 pairs for seeds 42, 43 and 44, alternating
their order. All six reports and failures are retained. The shared runner defaults
to one Inductor compiler worker; an explicit `TORCHINDUCTOR_COMPILE_THREADS`
overrides this, and reports record the environment setting.

QT plus real-data Actions runs include this profile alongside the original
small-batch pair. Summaries show median training-phase time from epoch 3 onward
separately from total training-call wall time. The former includes loading,
preprocessing and batch logging, excludes validation/figures/checkpoints, and
may still contain later compilation. Compiler caches are not cleared between
runs: neither column establishes fresh-cache performance. Real-data completion
still has no quality acceptance threshold; inspect accuracy for every seed.

### CUDA graph comparison

```bash
CUDA_VISIBLE_DEVICES=0 BENCHMARK_DATA_ROOT=examples \
  bash dev/check-benchmarks.sh qt-cudagraphs /tmp/qt-cudagraphs
```

This repeats the three-seed, batch-512, 60-epoch MNIST comparison with
`--compile-mode reduce-overhead` for both float and INT8. All other settings and
alternating execution order match `qt-large-batch`. Both profiles remain in the
optional QT plus real-data GPU workflow, with summaries and retained artifacts.
Explicit modes are recorded in JSON reports; process failures retain the exact
arguments. Neither a mode flag nor a completed real-data run guarantees CUDA
graph replay, convergence equivalence or a speedup. Compare all timings and
memory against float under the same mode, rather than an older float baseline.

### Model and optimizer graph comparison

```bash
CUDA_VISIBLE_DEVICES=0 BENCHMARK_DATA_ROOT=examples \
  bash dev/check-benchmarks.sh qt-optimizer-cudagraphs /tmp/qt-optimizer-cudagraphs
```

This adds `--optimizer-cudagraphs` to the same three-seed, batch-512, 60-epoch
comparison for both float and INT8. The optional QT plus real-data GPU workflow
runs it alongside the existing profiles, publishes its summary, and retains
reports and failures for 90 days. Keep the older profiles as controls: optimizer
graph replay is opt-in and does not improve every workload. See the
[measured larger-batch results](../../docs/benchmarks.md#larger-batch-model-and-optimizer-graph-results).

## Portable ONNX inference measurements

`onnx_inference` runs without importing PyTorch or `mini_trainer`. Use a prepared
NumPy, ONNX and ONNX Runtime environment appropriate to the target CPU/GPU; it
never installs packages. The repository's export extra supplies the CPU provider.
GPU provider installation is a separate environment choice, not an automatic
replacement of that installation.

Prepare a fixed `.npz` containing preprocessed arrays keyed by exact ONNX input
names (for the default exporter, `images`). Carry the preprocessing recipe, sample
identities, split manifest and labels alongside it. Use the same input file for
every recipe and machine. A batch is fixed by its array shape; prepare separate
files for batch-one edge latency and larger batch throughput measurements.
Copy each whole ONNX bundle, including external weights, to the target machine.

```bash
.venv/bin/python -m dev.benchmarks.onnx_inference \
    --model exported-float/model.onnx --model exported-int8/model.onnx \
    --inputs preprocessed-batch.npz --output /tmp/onnx-cpu-run-1 \
    --provider CPUExecutionProvider --threads 1 --warmup 3 --repeats 11

# In a separately prepared ONNX Runtime GPU environment:
python -m dev.benchmarks.onnx_inference \
    --model exported-float/model.onnx --model exported-int8/model.onnx \
    --inputs preprocessed-batch.npz --output /tmp/onnx-cuda-run-1 \
    --provider CUDAExecutionProvider --provider-options '{"device_id": "0"}'
```

Each output directory must be new. Reports include graph/external-weight/input
hashes, runtime versions/build, runner source hash, input shapes/dtypes, requested
and effective provider configuration, every timing sample and medians. Outputs
are saved by ONNX output name, suitable for a separate quality evaluation with
`mini_metrics` using the original class mappings and labels. This runner does not
infer score semantics, calibrate thresholds or impose a quality acceptance gate.

Profiling uses a separate session and records actual operation/provider counts,
including CPU execution in a GPU-requested run. An unavailable provider or a graph
that executes no operations on the requested provider fails instead of reporting
a CPU run as a GPU measurement. Partial CPU execution is reported, not prohibited:
shape and other auxiliary operations may legitimately use CPU. Inspect the profile
for the expensive operations. Integer weights or QDQ nodes alone are not proof of
integer execution. Use repeatable `--require-provider-op` arguments to require
particular **profiled runtime operation types** on the requested provider. Every
required type must occur in each supplied model and every occurrence must execute
on that provider; missing types and partial fallback fail before timing. Inspect
the profile first: graph optimizations can change operation names or fuse nodes.
Auxiliary CPU operations remain allowed when they are not among the requirements.

For example, add `--require-provider-op Conv --require-provider-op MatMulInteger`
to a native QT model invocation to require both the backbone and integer head on
CUDA. This currently **fails** on the tested ONNX Runtime 1.29.0 CUDA build: its
native integer head executes on CPU. The calibrated QDQ artifact instead runs
floating Conv/Gemm on CUDA, so requiring integer operation types also fails.
See the [measured placement results](../../docs/benchmarks.md#onnx-cuda-provider-placement).
A successful default measurement means some work ran on the requested provider;
it is not an integer-kernel or quality acceptance gate.

`--optimization disable|basic|extended|all` selects and records ONNX Runtime's
graph optimization level for both the profiling and timing sessions (`all` is the
unchanged default). This matters for provider partitioning: the tested TensorRT
QDQ graph acquired unsupported INT32 bias dequantization under the default
rewrites. With `disable`, its flat model ran as one TensorRT partition without
CPU operations. This setting does not disable TensorRT's own engine optimization.

Failures after output creation retain a failed `report.json`;
input/environment preflight failures leave no result directory.

Timing covers warm `Session.run` with CPU NumPy inputs and outputs, including
host/device transfers on GPU. It excludes image IO, preprocessing, export, session
construction and profiling. It is neither GPU-only kernel time nor end-to-end
image prediction latency. Sessions for all supplied models coexist during timing;
use individual model invocations if residency exceeds device capacity, and record
that different protocol. For repeated-process evidence, run at least three fresh
invocations with separate output directories and reverse model argument order in
alternate invocations. Avoid concurrent training, tests or other benchmarks.

Start with the exact same bundles on the local CPU, then repeat on Raspberry Pi
(or the intended ARM device) and the intended ONNX GPU device. Do not reuse
hardware-specific optimized graph caches across devices. Record the device model,
power/thermal configuration and concurrent workload alongside the report. The
unsigned activation recipe measured on x86 is a candidate to test, not a universal
GPU/ARM recipe. This inference runner does not validate HPC PyTorch quantized
training, native QT checkpoint export or million-class capacity.

### Maintained ONNX calibration command

`dev.benchmarks.onnx_calibration` creates a fresh QDQ model from ordered,
preprocessed NPZ batches. Use an explicitly prepared ONNX/ONNX Runtime environment;
the command installs nothing and requires the calibration-cache API tested with
ONNX Runtime 1.29.0. Calibration and the final smoke execution run on CPU with
`--threads 1` by default, including the first session created by the calibrator.

Supply a manifest such as:

```json
{
  "schema_version": 1,
  "split": "train",
  "batch_input": "images",
  "provenance": {
    "dataset": "versioned dataset manifest and hash",
    "checkpoint": "source checkpoint hash",
    "selection": "sample selection algorithm and seed",
    "preprocessing": "exact resize, channel order, normalization and dtype"
  },
  "batches": [
    {"path": "batch-000.npz", "sample_ids": ["train/image-001", "train/image-002"]}
  ]
}
```

Each NPZ must contain all model inputs under their ONNX names, with finite arrays
and matching dtypes/shapes. `batch_input` identifies the input whose leading
dimension matches `sample_ids`; other inputs can be static or unbatched. Paths
are relative to the manifest. Optional batch `sha256` values are checked before
use, and the command always records hashes of the bytes actually loaded. Sample
IDs must be unique across batches. The split must be `train` or `calibration`;
this declaration cannot prove separation from held-out data, so review the
selection and provenance. Preserve batch order and boundaries for reproducible
histogram collection. Prepare inputs using the source model's preprocessing;
this command does not infer image transforms or class mappings.

```bash
# Unsigned activations, signed per-channel weights, quantized biases (CPU candidate):
python -m dev.benchmarks.onnx_calibration \
    --model exported-float/model.onnx --manifest calibration/manifest.json \
    --output /tmp/qdq-cpu-1 --threads 1

# Signed symmetric activations and floating biases (tested TensorRT candidate):
python -m dev.benchmarks.onnx_calibration \
    --model exported-float/model.onnx --manifest calibration/manifest.json \
    --output /tmp/qdq-trt-1 --threads 1 \
    --activation-type int8 --symmetric-activations --float-bias
```

Defaults are Percentile 99.9, asymmetric histogram collection, symmetric INT8
per-channel weights, and Conv/Gemm/MatMul selection. `--method minmax`,
`--percentile`, `--symmetric-calibration`, `--per-tensor-weights` and repeated
`--op-type` flags make alternatives explicit. Histogram symmetry and activation
quantizer symmetry are separate settings: the tested TensorRT recipe derives
symmetric quantizer ranges from asymmetric histograms, so it does **not** use
`--symmetric-calibration`.

A new output directory receives a private source-model snapshot, augmented
calibration graph, `ranges.json`, quantized `model.onnx` and its external weights,
`calibration-smoke.npz`, and `report.json`. Source/external-file hashes, recipe,
versions, ordered sample IDs, input hashes and failures after output creation are
retained. Existing output directories are refused. The private snapshot isolates
ORT's shape-inference sidecars from the source directory. Histogram collection
retains raw activations for one batch at a time; model loading, histograms and
graph construction still consume memory. The artifact directory includes research
intermediates, not only deployment files.

The smoke run checks finite outputs on one calibration batch. It does not measure
held-out quality, integer execution coverage, speed or memory benefit. Continue
with provider/engine inspection and paired evaluation using mini_metrics. CPU
regressions cover both calibration methods, signed/unsigned recipes, multiple
inputs, thread limits, unchanged source files and retained failure reports in
`tests/test_benchmark_calibration.py`.

### TensorRT calibration candidate

The local candidate uses signed, symmetric INT8 activation/weight QDQ, per-channel
weights and floating biases. It derives symmetric ranges from the retained
Percentile 99.9 training-split calibration cache. With ONNX Runtime 1.29.0's
`quantize_static`, the relevant arguments are `quant_format=QuantFormat.QDQ`,
`activation_type=QuantType.QInt8`, `weight_type=QuantType.QInt8`, `per_channel=True`
and `extra_options={"ActivationSymmetric": True, "WeightSymmetric": True,
"QuantizeBias": False}`. The maintained calibration command above regenerates
this recipe from reviewed input batches into separate artifacts. This changes
the quantization recipe; it does not
preserve the native training export's dynamic quantizer.

In a separately prepared compatible TensorRT/ONNX Runtime GPU environment:

```bash
python -m dev.benchmarks.onnx_inference \
    --model signed-symmetric-qdq/model.onnx --inputs preprocessed-batch.npz \
    --output /tmp/trt-placement-1 --provider TensorrtExecutionProvider \
    --optimization disable \
    --provider-options '{"trt_max_workspace_size":"1073741824","trt_engine_cache_enable":true,"trt_engine_cache_path":"/tmp/trt-cache-1"}'
```

TensorRT profiles show fused partition names rather than individual Conv/Gemm
operations. A partition on GPU alone does not prove integer arithmetic: inspect
engine layer input/weight types and tactics using detailed TensorRT engine
inspection. Direct hierarchical engine construction requires registering the
standard TensorRT plugins (`init_libnvinfer_plugins`) for scatter reductions.
Do not copy device-specific engines between target machines. See the
[TensorRT evidence and quality limits](../../docs/benchmarks.md#tensorrt-int8-calibration-candidate).

### Maintained TensorRT build and inspection command

`dev.benchmarks.tensorrt_build` builds an engine from an existing ONNX graph,
records detailed layer inspection, executes one supplied input, and optionally
checks its outputs against a named reference NPZ. It uses CUDA PyTorch for probe
buffers/stream management, plus explicitly installed compatible TensorRT and ONNX;
it never installs packages. This is a developer validation command, not a
PyTorch dependency for the deployed TensorRT engine. Module import and `--help`
do not load TensorRT or PyTorch.

```bash
python -m dev.benchmarks.tensorrt_build \
    --model signed-symmetric-qdq/model.onnx --inputs preprocessed-batch.npz \
    --profiles profiles.json --output /tmp/trt-build-1 --optimization 3

# Practical floating baseline from the unquantized graph:
python -m dev.benchmarks.tensorrt_build \
    --model exported-float/model.onnx --inputs preprocessed-batch.npz \
    --profiles profiles.json --output /tmp/trt-fp16-build-1 --fp16 --optimization 3
```

Every input name must occur in the profiles JSON, for example:

```json
{"images": {"min": [1, 3, 128, 128], "opt": [8, 3, 128, 128], "max": [8, 3, 128, 128]}}
```

Without `--profiles`, all min/opt/max shapes equal the supplied input shapes.
The smoke-test input may be anywhere within an explicit profile. Fixed network
dimensions and input dtypes must match; arrays are not implicitly cast. Profiles
control the engine's supported input range, not a claim that every shape in that
range has been executed or validated. No model/head allowlist is used.

The default workspace limit is 1024 MiB and builder optimization level is 3;
`--workspace-mib`, `--optimization` and `--device` are explicit controls. TF32 is
disabled unless `--tf32` is supplied. `--fp16` allows FP16 tactics. QDQ precision
comes from the graph; the command does not calibrate, insert QDQ or replace the
native training quantizer. Standard TensorRT plugins are registered before parsing.

A new output directory receives `model.engine`, `layers.json`, `outputs.npz` and
`report.json`, including graph/external-weight/input hashes, SDK versions, GPU,
settings, engine hash/size and reported context-memory requirement. Failures after
output creation retain diagnostics, including parser errors and warning/error
messages. An existing directory is never overwritten. A built engine or smaller
file does not certify integer execution, speed, total GPU-memory reduction or
acceptable model quality; inspect layer types/tactics and run the separate studies.

Optional `--reference expected-outputs.npz` requires exact output names, shapes
and dtypes. Floating parity defaults to `--rtol 1e-4 --atol 1e-5`; integer/bool
outputs compare exactly. Failed parity retains the engine, actual outputs and
comparison report. Absence of a reference is recorded and means parity was not
checked. A single input does not replace full dataset evaluation with mini_metrics.

The probe currently supports ordinary execution inputs and linear device IO.
Shape-tensor input profiles, host/nonlinear IO bindings, unresolved data-dependent
output dimensions and empty outputs fail explicitly. These are probe limitations,
not blanket limitations of ONNX or TensorRT. Engines must be rebuilt and tested
for their actual destination device/software environment.

CPU profile/reference contracts run in the normal suite. In the prepared GPU
environment, set `CUDA_VISIBLE_DEVICES` and `RUN_CUDA_TESTS=1`, then run
`bash dev/check.sh test tests/test_benchmark_tensorrt.py` to include real engine
construction, successful reference matching, retained mismatches and parser
failure diagnostics.

### Maintained paired TensorRT timing command

`dev.benchmarks.tensorrt_pair` compares two existing engines on the same named
NPZ input arrays. It has no model/head or batch-size assumptions; prepare a
separate NPZ for each workload shape, including all static inputs. Both engines
must accept the same input dtypes/shapes in profile 0 and produce the same output
names/shapes. Output precisions may differ and are recorded. It uses the explicit
TensorRT/CUDA PyTorch environment from the build command and installs nothing.

```bash
OMP_NUM_THREADS=1 python -m dev.benchmarks.tensorrt_pair \
    --baseline fp16/model.engine --candidate int8/model.engine \
    --inputs preprocessed-batch.npz --output /tmp/pair-1
OMP_NUM_THREADS=1 python -m dev.benchmarks.tensorrt_pair \
    --baseline fp16/model.engine --candidate int8/model.engine \
    --inputs preprocessed-batch.npz --output /tmp/pair-2 --reverse
OMP_NUM_THREADS=1 python -m dev.benchmarks.tensorrt_pair \
    --baseline fp16/model.engine --candidate int8/model.engine \
    --inputs preprocessed-batch.npz --output /tmp/pair-3
```

Run processes sequentially, with no concurrent benchmarks, tests or training.
Record power/thermal settings and concurrent system load alongside the reports.
Use matched builder settings and practical baselines for the target hardware;
these commands do not infer whether an engine uses INT8, FP16 or another precision.
Rebuild engines for each destination device. Repeat batch sizes/resolutions and
10k/100k-class local head workloads explicitly; reserve million-class full models
for suitably sized systems.

Defaults are 10 warmup pairs and 31 measured pairs. `--warmup`, `--repeats`,
`--reverse` and `--device` are explicit controls. Each pair runs baseline and
candidate adjacently; their order alternates, and `--reverse` reverses that order.
The summary reports the median of **paired candidate/baseline ratios**, where
less than one means lower candidate latency. Retain per-process results rather
than treating repetitions from one process as independent device replications.

Host wall time covers input copies to GPU, execution, output copies to CPU and
stream synchronization, with preallocated buffers and a nondefault CUDA stream.
Default host buffers are pageable; `--pinned` uses pinned buffers and nonblocking
copies, with the same final synchronization. Compare these as separate IO regimes.
No CUDA-event timing is used: earlier WSL event readings were inconsistent with
enclosing host durations. This command excludes preprocessing, allocations,
shape changes, engine loading and construction from steady-state measurements.
It does not use CUDA graph capture. File reading and deserialization/context
creation have separate observations, but these are not whole-process cold-start
measurements and may benefit from caches.

Both engines, contexts and IO buffers coexist. Engine file sizes and reported
context-memory requirements are recorded, **not total runtime memory savings**.
Profile shape tensors, nonlinear/host engine IO and unresolved/empty outputs are
explicitly unsupported by this probe. This restriction concerns engine bindings;
the command's host transfer buffers are separate from those device bindings.

A new directory receives `report.json`, raw paired host durations, engine/input
hashes, versions, GPU/thread information and final named output NPZs. Completed
pairs and diagnostics survive later failures; existing output directories are
refused. Final outputs must be finite, but this is not quality or numerical-parity
acceptance. Use the separate full-dataset mini_metrics evaluation and engine
inspection to establish the quality/efficiency trade-off. CPU tests cover pairing,
summary arithmetic and invalid clocks; optional GPU tests cover actual multi-input
execution, both transfer modes and retained failures in
`tests/test_benchmark_tensorrt_pair.py`.
