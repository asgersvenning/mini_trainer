# Quantized training benchmarks

For a target-machine handoff, start with the
[representative paired EfficientNetV2 profile](#representative-paired-efficientnetv2-training-profile).
The smaller probes below isolate specific bottlenecks; current conclusions and
priorities live in the [branch roadmap](../../docs/quantization-roadmap.md).

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

Add `--pretrained --fine-tune` to both precision runs for the existing builder's
parameter-frozen fine-tuning regime. The benchmark keeps floating backbone
parameters in FP32 and uses the requested AMP dtype for compute; INT8 preparation
still follows the ordinary recipe and reports its actual operator coverage.
Backbone parameters receive no optimizer updates, but BatchNorm running statistics
and dropout retain normal training behavior. This is distinct from the
`large_head_training --frozen` capacity probe, which evaluates the backbone.
Reports record `fine_tune`, `backbone_floating_dtype` and `backbone_training_mode`;
the option is also retained in failure reports. Apply identical settings and seeds
to both precision runs. The flag alone does not establish a speed, memory or
quality benefit, and random frozen features are only an execution diagnostic.

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

[Local measured results](../../docs/archive/benchmark-history.md#integrated-int8-training) include
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
See the [reader findings](../../docs/archive/benchmark-history.md#uint8-nearest-resize-in-the-streaming-reader).

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
[measured larger-batch results](../../docs/archive/benchmark-history.md#larger-batch-model-and-optimizer-graph-results).

## Training predictions to paired quality evaluation

Use saved `dev.benchmarks.run` output directories to prepare the same held-out
contract consumed by `quality_compare`, independently of the training device:

```bash
.venv/bin/python -m dev.benchmarks.training_predictions \
  --baseline results/float --candidate results/int8 --output results/quality-inputs
# Use an explicitly prepared environment containing mini_metrics:
/path/to/metrics-python -m dev.benchmarks.quality_compare \
  --manifest results/quality-inputs/manifest.json --output results/quality
```

Run from the repository root. This installs nothing and does not import PyTorch,
load checkpoints or rerun inference. The adapter accepts the runner's synthetic
and real-dataset layouts, flat heads and any number of reported hierarchy levels.
Transfer `report.json`, `predictions.npz` and the dataset manifest from the training
machine; checkpoints and source images are not needed for this evaluation step.
The training report's checkpoint identifier is retained, not independently
verified against a checkpoint file.

Both runs must have completed training, checkpoint reload and inference, and
must declare that test images were not used for training or selection. A synthetic
run that completed inference but missed its quality gate can still be evaluated;
its original failure status remains in the manifest. Execution failures without
completed inference are rejected. The adapter checks each dataset-manifest hash,
held-out paths/labels, ordered class mappings, image hashes, archive levels,
leaf aliases and finite floating scores before creating output. Different
manifest order or training metadata is allowed when the held-out identities and
image hashes agree; different test labels, image hashes or class order is rejected.

The new directory contains canonical `baseline.csv`, `candidate.csv` and
`manifest.json`, with source hashes and both original training reports. Fixed
argmax predictions use row-wise softmax confidence at threshold zero, without
abstention or threshold optimization. Confidence conversion does not allocate a
second dense float64 score matrix, although NumPy still loads an archive's score
array into memory. The confidence values are not calibration evidence. The
existing evaluator computes Macro-F1, Macro-Recall, Macro-Precision, Coverage and
Theil's U independently per level, and records its mini_metrics source hashes.

This is a saved-prediction quality comparison. It does not establish runtime
performance, checkpoint-to-score parity, source-image integrity beyond the
recorded hashes, or production acceptance. Preserve the evaluator's report and
CSV/manifest bundle alongside the original training reports for continuous runs.

### Epoch statistics for convergence comparisons

The dataset runner retains the existing `MetricLogger` so
`training/logs/summary.csv` records actual train/validation statistics for every
epoch, including per-level hierarchical losses and accuracies. These are the
logger's unweighted means of batch statistics, not the held-out mini_metrics
Macro-F1/Recall/Precision/Coverage/Theil's U results. Use the paired prediction
adapter and quality evaluator for those metrics.

Earlier runs through `aaa90a1` passed `logger_cls=[]`: their CSV statistics are
zero placeholders and cannot support convergence claims, and their `best.pt`
selection saw a constant validation statistic. The dataset runner explicitly
reloads `last.pt` for held-out predictions, so those independently calculated
quality results and the recorded timing/memory measurements remain valid within
their stated scope. Do not treat the historical `best.pt` files as validated
best-epoch choices. Restoring statistics changes logging overhead; rerun both
precisions together before comparing new timing results with one another.

### Representative paired EfficientNetV2 training profile

`qt-efficientnet` composes the existing dataset runner, training-prediction adapter
and five-metric evaluator. By default it trains pretrained EfficientNetV2-S with
symmetric normalized flat and hierarchical heads, full and parameter-frozen
backbones, seeds 42/43/44, BF16 and native INT8: 24 sequential training processes
and twelve held-out comparisons. It uses the reviewed Blair taxonomy, five epochs,
batch 32, 128px images, MuonAuxAdamW at learning rate 0.01, CPU caching and zero
workers. Model/optimizer compilation is disabled. Odd seeds reverse precision and
full/frozen ordering. Metric evaluation starts after every training process has
finished, preserving isolation from that evaluation workload.

Use explicitly prepared training and mini_metrics environments; the command
installs nothing and does not assume a sibling checkout. The metrics interpreter
is checked before training begins. Run from an otherwise idle allocation and
record its actual hardware/runtime in the retained reports:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=0 \
BENCHMARK_DATA_ROOT=/path/to/examples \
BLAIR_CLASS_SPEC=/path/to/reviewed/class_spec.json \
BENCHMARK_METRICS_PYTHON=/path/to/metrics-env/bin/python \
bash dev/check-benchmarks.sh qt-efficientnet fresh-results
```

The following environment variables select a bounded diagnostic or longer budget:

| Variable | Default | Accepted values |
| --- | --- | --- |
| `BENCHMARK_HEAD` | `both` | `both`, `flat`, `hierarchical` |
| `BENCHMARK_TRAINING_MODE` | `both` | `both`, `full`, `frozen` |
| `BENCHMARK_SEEDS` | `42 43 44` | Space-separated unique nonnegative integers |
| `BENCHMARK_EPOCHS` | `5` | Positive integer; each model starts fresh with this schedule budget |
| `BENCHMARK_PYTHON` | `.venv/bin/python` | Existing training interpreter |

For example, set `BENCHMARK_HEAD=hierarchical`, `BENCHMARK_TRAINING_MODE=full`,
`BENCHMARK_SEEDS=42` and `BENCHMARK_EPOCHS=20` for a fixed-seed longer-budget pair.
This is a diagnostic, not a replacement for repeated-seed qualification. Choose
the budget before examining held-out results; the command always evaluates final
`last.pt` checkpoints and does not select a seed or threshold on the test split.

The results directory must be new. It retains each training report, epoch CSV,
checkpoints, predictions, pair input manifests, mini_metrics reports and logs.
`summary.md` contains both training measurements and per-level quality differences
for Macro-F1, Macro-Recall, Macro-Precision, Coverage and Theil's U. Differences
are multiplied by 100, including Theil's U; undefined metrics remain explicit.
Training or pairing/evaluation failures make the command return nonzero and are
retained in reports and the summary. Other pairs still run so failures cannot
silently remove difficult configurations from the evidence.

The default matrix is substantially larger than the TinyConv `qt-real` profile.
Size the allocation and artifact storage accordingly. This command is suitable
for a configured target runner but is not yet wired into the scheduled GPU job;
durable result hosting and joint quality/resource acceptance gates remain open.
Local CUDA success does not establish HPC throughput, desktop ONNX performance
or ARM inference support.
