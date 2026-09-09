# ONNX deployment benchmarks

Start with the composed [CPU comparison](#composed-cpu-deployment-comparison) or
[TensorRT evaluation](#composed-tensorrt-deployment-evaluation). The remaining
commands prepare their inputs or diagnose an individual stage. Follow the
[branch roadmap](../../docs/quantization-roadmap.md) before expanding experiments.

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
See the [measured placement results](../../docs/archive/benchmark-history.md#onnx-cuda-provider-placement).
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
`tests/benchmarks/test_benchmark_calibration.py`.

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
[TensorRT evidence and quality limits](../../docs/archive/benchmark-history.md#tensorrt-int8-calibration-candidate).

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
`bash dev/check.sh test tests/benchmarks/test_benchmark_tensorrt.py` to include real engine
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
`tests/benchmarks/test_benchmark_tensorrt_pair.py`.

### Maintained paired quality comparison

`dev.benchmarks.quality_compare` evaluates baseline and candidate prediction CSVs
using mini_metrics Macro-F1, Macro-Recall, Macro-Precision, Coverage and Theil's U
independently at each level. It uses an explicit held-out manifest to check sample
identity, labels, complete level coverage and declared class mappings. It does
not run model inference; connect the CSVs produced by a dataset inference pass
to this contract. This format avoids requiring complete score matrices merely
to compare classification metrics for large heads.

Prepare mini_metrics explicitly in the evaluation environment. The command does
not install it or assume a sibling checkout. It records the imported package path
and Python source hashes separately from installed distribution metadata: an
explicit source checkout on `PYTHONPATH` can differ from the installed release.

```bash
PYTHONHASHSEED=0 OMP_NUM_THREADS=1 python -m dev.benchmarks.quality_compare \
    --manifest heldout-comparison.json --output /tmp/quality-1
```

The CSV columns are exactly `instance_id,filename,level,label,prediction,confidence,threshold`.
IDs are nonnegative INT64 integers; levels are zero-based in manifest order.
Class names remain literal strings, including numeric-looking names such as
`001`. Confidence must be finite in [0,1] and threshold must be zero. Rows may
arrive in any order; duplicate/missing samples or levels, changed filenames/labels,
unknown predictions and hash mismatches are rejected. Optional derived metric
columns are deliberately excluded so they cannot override the evaluated labels
or prediction policy.

A minimal manifest has this structure (include every evaluated sample):

```json
{
  "schema_version": 1,
  "split": "val",
  "provenance": {"dataset": "manifest hash and split selection", "preprocessing": "exact inference transforms"},
  "levels": [{"name": "leaf", "classes": ["cat", "dog"]}],
  "samples": [
    {"instance_id": 0, "filename": "cat.jpg", "labels": ["cat"]},
    {"instance_id": 1, "filename": "dog.jpg", "labels": ["dog"]}
  ],
  "baseline": {
    "path": "baseline.csv", "classes": [["cat", "dog"]],
    "provenance": {"model": "checkpoint/export/engine hashes", "preprocessing": "baseline transforms"}
  },
  "candidate": {
    "path": "candidate.csv", "classes": [["cat", "dog"]],
    "provenance": {"model": "checkpoint/export/engine hashes", "preprocessing": "candidate transforms"}
  }
}
```

Paths are relative to the manifest. Baseline/candidate may each include `sha256`
to require exact input bytes. Each artifact's `classes` must match the ordered
level mappings used to translate its score columns to labels. These are explicit
declarations; the evaluator cannot prove how an external producer translated
scores or whether a declared `val`/`test` split is independent of training and
calibration. Review the dataset, preprocessing and model provenance. Hierarchical
models add levels and one label per level to each sample; this command computes
ordinary per-level metrics, not hierarchical path metrics.

Policy is fixed: `optimal=False`, threshold zero, no abstention, `known_only=False`,
`per_class=False`, and an explicit `opt_crit=MacroF1`. No evaluation subset is
removed for threshold tuning. Coverage is therefore one for these complete,
known-label predictions; abstention/unknown-label policies need separate studies.
The package's own observed-class and undefined-value semantics are preserved;
declaring a class does not force it into mini_metrics' macro denominator.

The report retains both sets of metrics, candidate-minus-baseline differences in
their original units, prediction-change counts, provenance and canonical CSVs.
Multiply differences by 100 to express percentage-point changes where appropriate.
Undefined values become explicit JSON `null` entries and appear in
`undefined_metrics`; they are not silently replaced by zero. `status=evaluated`
means calculation completed, not that production quality is acceptable. Failures
after output creation retain completed results; existing directories are refused.

The tested mini_metrics F1 implementation iterates an unordered class set, so
last-bit summation differences can occur across Python hash seeds. Set
`PYTHONHASHSEED` before starting Python for repeatable runs, retain its recorded
value, and use a numerical tolerance when comparing independently recorded
floating results. Do not alter metric definitions to force exact historical
bytes. Correctness tests cover synthetic oracle metrics, literal class identities,
reordered predictions, invalid contracts and undefined Theil's U; tests requiring
mini_metrics skip explicitly when that optional evaluation package is absent.

### Maintained full-dataset prediction collection

`dev.benchmarks.dataset_inference` connects preprocessed held-out input batches to
the paired quality evaluator. ONNX Runtime supports the CPU/edge collection path
and explicit GPU providers; a separate TensorRT backend executes built engines.
The CPU path imports neither PyTorch, TensorRT nor mini_metrics. Prepare ONNX,
ONNX Runtime and NumPy explicitly for that path; TensorRT collection uses the
existing compatible TensorRT/CUDA PyTorch environment. The command installs nothing.

Extend the quality manifest's `schema_version`, `split`, `provenance`, `samples`
and `levels` fields with input batches and explicit output bindings. Baseline and
candidate artifact fields are produced by collection rather than supplied here:

```json
{
  "schema_version": 1,
  "split": "val",
  "provenance": {"dataset": "manifest hash and held-out selection", "preprocessing": "exact transforms"},
  "levels": [{"name": "leaf", "classes": ["cat", "dog"], "output": "output_0", "score_semantics": "logits"}],
  "samples": [
    {"instance_id": 0, "filename": "cat.jpg", "labels": ["cat"]},
    {"instance_id": 1, "filename": "dog.jpg", "labels": ["dog"]}
  ],
  "batch_input": "images",
  "batches": [{"path": "batch-000.npz", "sample_ids": ["0", "1"]}]
}
```

Each NPZ contains all named model inputs; static/unbatched inputs are permitted.
The leading dimension of `batch_input` must match the batch's `sample_ids`, which
are canonical string forms of the manifest's integer IDs. Batches must cover every
sample exactly once. Paths are relative to the manifest, optional batch `sha256`
values are checked, and hashes of the loaded bytes are always recorded. Batch
order may differ from sample declaration order. Different batch sizes, including
the final partial batch, must fit the runtime graph/engine profile.

Each mapped output must be finite floating scores of shape `[batch, classes]`.
Declare either `logits` (confidence computed by stable softmax) or `probabilities`
(checked to lie in [0,1] and sum to one). Class lists follow score-column order;
the collector cannot infer a trustworthy mapping from an opaque engine or prove
the input preprocessing matches the checkpoint. Bind these from reviewed export
metadata and preserve provenance. Add one mapping per hierarchical level; no
backbone/head allowlist is used.

```bash
# CPU/edge collection, using an explicit ONNX candidate:
OMP_NUM_THREADS=1 python -m dev.benchmarks.dataset_inference \
    --model calibrated/model.onnx --manifest heldout-inputs/manifest.json \
    --output /tmp/heldout-cpu-1 --threads 1

# GPU baseline, then candidate and a ready-to-evaluate comparison manifest:
OMP_NUM_THREADS=1 python -m dev.benchmarks.dataset_inference \
    --backend tensorrt --model fp16/model.engine \
    --manifest heldout-inputs/manifest.json --output /tmp/heldout-fp16-1
OMP_NUM_THREADS=1 python -m dev.benchmarks.dataset_inference \
    --backend tensorrt --model int8/model.engine \
    --manifest heldout-inputs/manifest.json --output /tmp/heldout-int8-1 \
    --baseline-bundle /tmp/heldout-fp16-1/evaluation.json
PYTHONHASHSEED=0 OMP_NUM_THREADS=1 python -m dev.benchmarks.quality_compare \
    --manifest /tmp/heldout-int8-1/comparison.json --output /tmp/heldout-quality-1
```

ONNX collection defaults to `CPUExecutionProvider`, one intra-op thread, one
inter-op thread and full graph optimization. `--provider`, JSON
`--provider-options` and `--optimization disable` support explicit alternatives.
The requested provider must be available and activated, but registered providers
do not establish per-operator placement; use the maintained placement/inspection
commands separately. CPU fallback within a GPU partitioned graph is not ruled out.
TensorRT uses `--device` and profile 0, with standard plugins registered. Its
current linear device-IO and shape-tensor limitations match the paired runner.

The new output directory receives `predictions.csv`, `evaluation.json` and
`report.json`. Supplying `--baseline-bundle` additionally creates `comparison.json`
after checking matching split, samples, labels and ordered level mappings.
Different backend artifacts can be compared this way. CSV paths in generated
bundles are absolute; update them if moving artifacts, retaining their hashes.
`--save-scores` retains each batch's named arrays and hashes for numerical
reproduction checks. Without it, only predictions and provenance are retained,
avoiding a full-dataset score matrix for large heads.

Collection holds one batch's inputs/outputs at a time and streams CSV rows. It is
a correctness/quality path with per-batch buffer allocation, not an inference
performance benchmark. Confidence calculation may use higher precision than a
previous producer; raw scores and discrete predictions are separate contracts.
Failures retain completed batches and partial CSVs, while incomplete collection
does not publish an evaluation bundle. Existing directories are refused. Tests
cover multiple inputs, mapped levels, changing batch sizes, full CPU collection
through mini_metrics, saved scores, failures and CPU dependency isolation; the
intentional GPU test checks both batch sizes against CPU outputs.

### Maintained image input preparation

`dev.benchmarks.prepare_inputs` replaces the local image-to-NPZ preparation
scripts. It reads a version-1 benchmark dataset manifest and an ONNX export
manifest, checks class ordering and source image hashes, and writes the calibration
or held-out input contract used by the commands above. It uses the existing
repository image reader on CPU with **zero workers** and an explicit thread limit.

```bash
# Reproduce the training-split calibration selection:
OMP_NUM_THREADS=1 python -m dev.benchmarks.prepare_inputs \
    --dataset-manifest benchmark/dataset_manifest.json --data-root examples/blair \
    --export-manifest exported-float/manifest.json --output /tmp/calibration-inputs-1 \
    --split train --count 128 --seed 42 --batch-size 8 --score-semantics logits

# Complete held-out validation input set, without subsampling:
OMP_NUM_THREADS=1 python -m dev.benchmarks.prepare_inputs \
    --dataset-manifest benchmark/dataset_manifest.json --data-root examples/blair \
    --export-manifest exported-float/manifest.json --output /tmp/validation-inputs-1 \
    --split val --batch-size 8 --score-semantics logits
```

Pass the resulting `manifest.json` to `onnx_calibration` for `train` or
`dataset_inference` for `val`/`test`. Omitting `--count` preserves the complete
split in dataset-manifest order. Supplying it uses
`random.Random(seed).sample(split_records, count)` with explicit cardinality
checks. IDs are assigned in selected-record order; filenames and labels preserve
source identity. The full source records, their hashes and selection are retained
in `report.json`. Declared cross-split duplicate image hashes are rejected, and
each selected source image is rehashed before and after preparation. This does
not independently re-inventory unselected files or prove the supplied split policy.

The source manifest is the existing benchmark format: `class_spec.cls2idx` is a
flat mapping or numbered level mappings, and each record includes `path`, `split`,
`targets` (one integer per source level) and `sha256`. Paths must resolve under
`--data-root`. Export metadata must contain matching class indices, the image
input shape/dtype, ordered outputs, and the image-reader `resize_size`. Ambiguous
classifier metadata requires `--classifier-module`. Default source levels are
the first N levels for N exported levels; `--source-levels` selects others
explicitly. Class ordering must match exactly. `--level-names` replaces the default
`level_0`, `level_1`, ... names.

`--score-semantics` is required, because arbitrary exported forward outputs do
not establish whether they are logits or probabilities. Supply one value for all
levels or one per level. The exporter must declare preprocessing outside the
graph. Prepared batches must match its floating image-input shape and dtype,
including fixed batch dimensions; a partial batch is rejected if the graph does
not support it.

The default factory is
`dev.benchmarks.prepare_inputs:repository_preprocess`. It uses the existing
architecture resolver and recorded resize/preprocessing dtype, with pretrained
weights disabled and local-only configuration loading where the backend supports
it. It does not load checkpoint tensors or construct the trained classifier head;
the architecture getter may still construct a backbone to resolve its transforms.
It does not interpret the export recipe's descriptive text as executable transforms.

For custom transforms/configuration, provide a trusted importable Python factory
with `--preprocess-factory package.module:function` and JSON `--factory-args`.
The factory receives classifier metadata as its first argument and returns a
callable from the repository's resized RGB uint8 batch to one image tensor.
The callable must preserve row identity. The default factory accepts `model_args`
inside factory arguments for existing backend options. Factory and transform
execution each receive the declared seed for Python, legacy NumPy and CPU Torch
RNGs, with caller RNGs restored; custom generators or external randomness remain
the factory's responsibility. Use deterministic inference preprocessing and verify
input hashes. Factory source hash, arguments, representation and Torch/NumPy
versions are recorded. Current architecture defaults are not proof that an
unrecorded custom training transform has been reconstructed correctly.

A new directory receives batch NPZs, `manifest.json` and `report.json`. Failures
after output creation retain diagnostics and any completed batches; existing
directories are refused. Preparation processes one batch at a time, but its
architecture loader and source inventory still consume memory. This image helper
currently produces one floating image input; the lower-level calibration and
collection manifests continue to support manually prepared multiple inputs.
The default factory requires the repository's PyTorch/Torchvision environment.
The resulting NPZs can be transferred to the lighter ONNX CPU collection environment
without reconstructing preprocessing on the edge device.

### Paired inference quality pipeline

`dev.benchmarks.inference_pair` composes the maintained dataset collectors and
`mini_metrics` evaluator. Each stage runs in a fresh process, sequentially, so
baseline and candidate runtimes do not coexist in GPU memory. It accepts the
same prepared held-out manifest for both models and retains prediction CSVs,
source hashes, runtime reports, subprocess logs, five-metric comparisons and a
`summary.md` suitable for a CI job summary. It installs nothing.

```bash
.venv/bin/python -m dev.benchmarks.inference_pair \
  --manifest prepared-val/manifest.json \
  --baseline float/model.onnx --candidate int8/model.onnx \
  --output paired-quality \
  --baseline-runtime '{"backend":"onnx","threads":1}' \
  --candidate-runtime '{"backend":"onnx","threads":1}' \
  --metrics-python /path/to/metrics-env/bin/python
```

Use `{"backend":"tensorrt","python":"/path/to/gpu-env/bin/python"}` for an
engine artifact, independently on either side. ONNX options include `provider`,
`provider_options`, `optimization` and `threads`; TensorRT accepts `device`.
`save_scores: true` retains batch score arrays when needed. Interpreter paths
must name actual executables; virtual-environment symlinks are preserved. All
input paths are resolved from the calling directory. Child commands run from
this checkout, inheriting the explicitly prepared environment, with
`PYTHONHASHSEED=0` for repeatable metric evaluation. No local sibling package is
required; install a compatible `mini_metrics` in the chosen metrics environment.

An existing output directory is rejected. A failed stage stops the pipeline,
returns a nonzero exit status and retains its log and the top-level failure
report; later stages do not run. Use a new output directory for retries. Reports
record metric deltas in units of 0–1, not percentage points, and preserve undefined
metrics. `evaluated` means collection and metric calculation completed, not that
the candidate passed a production gate.

This command composes **quality evaluation only**. Preparation, calibration,
engine builds, integer-placement inspection and paired performance measurement
remain separate maintained commands. In particular, collector process duration
is not a latency benchmark. Run the ONNX recipe on the actual ARM device and the
TensorRT recipe on the intended GPU before making target support claims.

### Single-process ONNX CPU memory probe

Use `dev.benchmarks.onnx_cpu_memory` on Linux, including the intended ARM device,
with **one model in each fresh interpreter process**. It measures a CPU-only,
unprofiled session; it does not import PyTorch or `mini_trainer`. NumPy, ONNX and
ONNX Runtime must already be installed for that architecture.

```bash
.venv/bin/python -m dev.benchmarks.onnx_cpu_memory \
  --model float/model.onnx --inputs batch-one.npz --output cpu-float-trial-1 \
  --threads 1 --warmup 3 --repeats 31
.venv/bin/python -m dev.benchmarks.onnx_cpu_memory \
  --model int8/model.onnx --inputs batch-one.npz --output cpu-int8-trial-1 \
  --threads 1 --warmup 3 --repeats 31
```

Repeat in new processes and new directories at least three times, reversing the
baseline/candidate order on alternate trials. Keep input hashes, thread counts,
CPU affinity and runtime settings matched. Run timing trials without competing
benchmarks. A fresh process does not imply cold filesystem caches; the command
neither evicts caches nor changes the machine's CPU governor, affinity or cooling.
For sustained edge claims, also run longer trials and record target power/cooling
conditions and throttling. The current command does not collect thermal telemetry.

Reports retain all warm timing samples, session-construction and first-inference
times, effective providers, CPU affinity, runtime versions/build, source/input
hashes and named output arrays. Memory snapshots cover runtime import, input
loading, session creation, first inference, warmup and the final measurement.
`resident_bytes` and `proportional_resident_bytes` use Linux `smaps_rollup` RSS/PSS;
`peak_resident_bytes` uses the approximate `status` VmHWM high-water mark since
exec. Shared mapped pages contribute fully to RSS and proportionally to PSS.
See the [kernel proc documentation](https://docs.kernel.org/filesystems/proc.html)
for these accounting distinctions and the accuracy limits of status counters.

The peak includes interpreter/import/input/output and validation allocations. It
is not a model-only allocation count, and subtracting two lifetime peaks does not
isolate model memory. `ru_maxrss` is deliberately not used: a local regression
showed that it retained a parent's pre-exec high-water mark in a fresh child,
while VmHWM described the child's new address space. Snapshot reads are outside
the inference timing intervals. ONNX provenance inspection and output serialization
happen after the last memory snapshot to avoid inflating the reported peak with
large inline graph tensors or archive-writing buffers.

An existing output directory is rejected; execution failures retain partial
reports. `measured` is not deployment acceptance. Pair results with the maintained
quality evaluator and separate operator-placement probe. Timing excludes image IO
and preprocessing; CPU/x86 measurements do not establish ARM kernels, memory,
quality or speed, and this probe does not measure GPU memory.

### CPU-specific QDQ candidate for EfficientNetV2

The native-checkpoint CPU follow-up found that the TensorRT-oriented signed,
floating-bias recipe left 107 convolutions floating under the tested x86 ONNX
Runtime. Quantizing biases alone did not change that coverage. The maintained
calibrator's unsigned-activation, INT32-bias defaults instead produced 170
`QLinearConv` and two `QGemm` executions for both normalized symmetric flat and
hierarchical EfficientNetV2-S heads in that environment.

```bash
.venv/bin/python -m dev.benchmarks.onnx_calibration \
  --model materialized/model.onnx --manifest prepared-train/manifest.json \
  --output cpu-qdq --activation-type uint8 \
  --method percentile --percentile 99.9 --threads 1

.venv/bin/python -m dev.benchmarks.onnx_inference \
  --model cpu-qdq/model.onnx --inputs batch-one.npz --output cpu-placement \
  --provider CPUExecutionProvider --threads 1 \
  --require-provider-op QLinearConv --require-provider-op QGemm
```

This recipe uses asymmetric activation ranges, symmetric per-channel INT8 weights
and quantized INT32 biases; omit `--symmetric-activations` and `--float-bias`.
It is a separate CPU deployment candidate, not a replacement for the signed
TensorRT recipe. Required operator names in the example reflect these tested
models; inspect the actual optimized execution for other architectures. The
presence of required integer types does not by itself prove that all weighted
operations use them: also inspect the counts and remaining floating operators.

Use the paired quality pipeline for the complete held-out split and the isolated
CPU memory probe in separate fresh processes for resource measurements. Keep the
same input files, explicit threads and floating baseline. This x86 result does
not establish ARM fusion, kernel behavior or performance; run the full procedure
on the intended edge device before accepting that profile. See the
[measured comparison](../../docs/archive/benchmark-history.md#cpu-specific-activation-and-bias-calibration).

### Composed CPU deployment comparison

`dev.benchmarks.cpu_deployment` connects full held-out quality evaluation,
operator/provider inspection and repeated isolated memory/latency trials for
already exported baseline and candidate ONNX bundles. Run from a checkout in a
prepared Linux CPU environment; dependencies are never installed implicitly.

```bash
.venv/bin/python -m dev.benchmarks.cpu_deployment \
  --baseline materialized/model.onnx --candidate cpu-qdq/model.onnx \
  --manifest prepared-val/manifest.json --inputs batch-one.npz \
  --output cpu-deployment-run-1 --threads 1 --trials 3 --warmup 3 --repeats 31 \
  --require-provider-op QLinearConv --require-provider-op QGemm \
  --metrics-python /path/to/metrics-env/bin/python
```

The manifest supplies held-out identities, class ordering and score semantics.
The separate NPZ selects the fixed batch shape for resource measurement; carry
its preprocessing and sample-selection provenance alongside it. Baseline and
candidate use that same NPZ. Preparation and calibration remain separate steps.
The main interpreter supplies ONNX CPU dependencies, while `--metrics-python`
can select an independently prepared mini_metrics environment.

Execution is sequential: paired quality, baseline/candidate placement, then
resource trials. Every profiling and resource command gets its own process;
profiling allocations cannot inflate the memory trial's high-water mark. The
first trial runs baseline then candidate, the next reverses the order, and so
on. Each resource pair must have matching settings and reported environments.
The graph/external-weight hashes must match those observed during quality
collection, and the resource input hash must remain unchanged through placement
and timing. Drift or a failed child stops the run and retains the failed phase,
logs and completed evidence. An existing output directory is rejected.

The output contains the child reports and logs, a top-level `report.json`, and
`summary.md` combining all five quality deltas with per-trial latency and memory
ratios. These ratios compare separate-process warm medians; they are not adjacent
per-inference pairs. All raw timings, startup observations and memory accounting
remain in the child reports. Requested candidate operation types must occur on
CPU, but this does not assert that every weighted operation is integer: inspect
`execution` counts and remaining floating operators. The operation requirements
are explicit and optional; no model-specific operator list is hardcoded.

Run the command without competing tests or benchmarks when using timing results.
It does not evict disk caches or control temperature, affinity or power states.
`evaluated` means the procedure completed, not that quality/resource trade-offs
passed production acceptance. For visible CI reporting, retain the complete
output directory even on failure and append its `summary.md` to the job summary
when present. Durable cross-run hosting, recipe acceptance gates, preprocessing
costs and actual target-device verification remain separate requirements.

### Large-head training capacity command

`dev.benchmarks.large_head_training` makes the earlier large-class probe reusable
on the intended training machines. It builds the actual backbone with a symmetric,
normalized flat or hierarchical head, then performs MuonAuxAdamW updates (eager by default)
through the existing builder, scaler and optimizer-step helper. It uses fixed
synthetic uint8 images and labels, not a convergence or dataset-quality benchmark.

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1 \
  .venv/bin/python -m dev.benchmarks.large_head_training \
  --classes 10000 --backbone efficientnet_v2_s --batch-size 32 --image-size 128 \
  --dtype float16 --seed 42 --warmup 3 --steps 5 --output capacity-float

# Same settings, separate fresh process, quantized weights/saved Linear inputs:
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1 \
  .venv/bin/python -m dev.benchmarks.large_head_training \
  --classes 10000 --backbone efficientnet_v2_s --batch-size 32 --image-size 128 \
  --dtype float16 --seed 42 --warmup 3 --steps 5 --quantized --output capacity-int8
```

Add `--hierarchical` to both commands for a synthetic parent taxonomy grouping
consecutive leaves in groups of 100. Add `--frozen` to both for head-only training:
backbone parameters are frozen and its modules use evaluation mode, while the head
trains. Parameters intentionally frozen by the classifier remain frozen. Backbone
forward computation still runs on every update; embeddings are not cached.
`--dtype bfloat16` selects a separate AMP profile; `float32` disables autocast.
Use `--device cpu --dtype float32` only for offline floating diagnostics. Native
INT8 training requires an accessible CUDA device.

Add `--compile-optimizer` to use the existing optimizer-compilation path, and
optionally `--optimizer-cudagraphs` to request optimizer CUDA graphs. The model
forward/backward remains eager. Apply the same execution settings to both
precisions; `--warmup 8 --steps 5` provides a bounded initial comparison that
includes the helper's real eager initialization update before compilation.
Graph settings alone do not establish actual replay or a performance benefit.
Reports retain both flags and the composite optimizer's applied step count.
Compare setup peaks, measured allocated peaks and reserved memory separately:
lower steady allocated memory does not establish a lower startup requirement,
and graph pools may increase reserved memory. The probe has no scheduler or
checkpoint/resume workload; those contracts remain covered by the integrated
runner and optimizer tests.

Use new output directories, matching settings, alternating float/INT8 order and
paired seeds. A separate CPU generator fixes inputs independently of quantization
setup RNG consumption; reports retain their hash. The report includes warmup and
measured losses, applied-update flags, trainable/gradient parameter counts,
unused trainable parameter names, quantization coverage and storage, warm median
update time and CUDA allocated/reserved peaks. Valid normalized heads can retain
unused BatchNorm parameters; these are reported rather than silently reclassified
or treated as missing gradients in an active branch. Measured skipped/nonfinite
updates or a head with no gradients fail, retaining the partial report.

Setup/warmup allocation and measured-phase allocation are recorded separately.
Kernel-cache state is not controlled; there is no cold-compilation speed claim.
The probe does not exercise the full trainer's loading, scheduler, checkpoint,
resume, augmentation, distributed or convergence behavior. Use the integrated
real-data profiles for those comparisons. Run target GPU generations separately;
a laptop result cannot certify A40/A100/B300 or the intended desktop. Keep full
models at 100k classes or below on the laptop, and increase capacity only on a
machine provisioned for it. No weights or datasets are saved by this probe.

### Isolated TensorRT memory command

Use `dev.benchmarks.tensorrt_memory` in a **fresh Linux process for each engine
and trial**, after preparing TensorRT/CUDA explicitly. The command installs
nothing and accepts the same arbitrary named NPZ inputs as the build and paired
latency commands. Engines must be rebuilt for the intended target environment.

```bash
CUDA_VISIBLE_DEVICES=0 python -m dev.benchmarks.tensorrt_memory \
  --engine results/fp16/model.engine --inputs inputs-8.npz \
  --output results/memory-fp16-trial1 --runs 20 --threads 1
CUDA_VISIBLE_DEVICES=0 python -m dev.benchmarks.tensorrt_memory \
  --engine results/int8/model.engine --inputs inputs-8.npz \
  --output results/memory-int8-trial1 --runs 20 --threads 1
```

Repeat in fresh processes and alternate engine order. Stop competing GPU jobs
before measuring. `--pinned` enables pinned host buffers and nonblocking copies;
keep this choice, input shapes and run counts matched across candidates.
`--device` selects the visible CUDA device, with engine optimization profile zero.
Each output directory must be new. Reports retain engine/input/output hashes,
versions, GPU identity, IO contracts, TensorRT warnings, and available snapshots
if deserialization, input validation or execution fails.

Snapshots cover host memory before runtime imports, then host/device memory after
CUDA initialization, engine load, context/IO allocation and synchronized repeated
execution. Every inference output must be finite. The command records Linux
RSS/PSS/swap and the approximate host high-water mark, device-wide free/total
memory, and PyTorch allocated/reserved counters separately. Snapshots precede
output serialization. Model decoding/preprocessing is external.

Interpret the fields according to their scope: device-wide changes can include
other processes and driver accounting; PyTorch counters omit TensorRT-owned
allocations; context requirements omit engine weights and other runtime costs.
These snapshots do **not** measure total transient GPU allocation peaks. Importing
PyTorch for CUDA/IO management also affects this process's footprint. Use paired
latency and full `mini_metrics` quality evaluation separately before accepting a
memory/quality trade-off. CPU/WSL evidence does not establish target GPU results.

### Composed TensorRT deployment evaluation

`dev.benchmarks.tensorrt_deployment` joins the maintained held-out collector and
`mini_metrics` comparison with build inspection, fresh-process paired latency and
isolated single-engine memory trials. Prepare the two engines on the intended
hardware with `dev.benchmarks.tensorrt_build`, then supply its **build directories**
containing `model.engine`, `report.json`, and `layers.json`:

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python -m dev.benchmarks.tensorrt_deployment \
  --baseline-build results/fp16 --candidate-build results/int8 \
  --manifest heldout/manifest.json --inputs heldout/batch-00000.npz \
  --output results/deployment-comparison \
  --metrics-python /path/to/metrics-environment/bin/python \
  --trials 3 --warmup 10 --repeats 31 --memory-runs 20 --threads 1
```

The active interpreter must have explicitly prepared TensorRT/CUDA dependencies;
the separate metrics interpreter must provide `mini_metrics`. Nothing is
installed or synchronized. The collector manifest declares preprocessing,
held-out sample identities, class order, output names and score semantics. Use
representative preprocessed timing inputs with shapes inside both engines'
profiles. Any number of declared classification levels is supported by the
existing collector contract; there is no backbone/head name allowlist.

Each build's engine and layer-inspection hashes must match its successful build
report. Quality collection runs first in fresh processes, followed by all five
metrics. Engine bytes, manifest bytes, ordered consumed batches and runtime
identity must agree before resource measurement begins. Each resource trial runs
adjacent paired latency in a new process, then one fresh memory process per
engine; trial order alternates. Resource input and engine hashes are checked
against the declared artifacts. `--device` and optional `--pinned` apply to both
resource runners. Memory uses the same NPZ inputs as timing and profile zero.

Retain the entire output directory: it contains stage commands/logs, report
hashes, canonical prediction CSVs, quality deltas and undefined metrics, raw
paired timings, memory stages, and `summary.md`. Keep the source build bundles
and input manifests/arrays with those artifacts for reproduction; the comparison
does not copy the source engines or datasets. Failed stages return nonzero and
retain the available reports instead of continuing into a misleading comparison.
The CLI summary can be included in a CI job summary and the directory uploaded
as an artifact; remote GPU orchestration and durable publication are separate.

`status=evaluated` means the protocol completed, **not** that the candidate passed
a production quality/resource threshold. Read the metric deltas and undefined
values together with measured benefit. The inspection inventory does not itself
prove a speedup or complete integer coverage. Host latency excludes preprocessing
and loading, and both engines coexist during timing. Memory comes from separate
processes; device-wide snapshots are not per-process or transient peak readings.
Use quiescent target hardware and record deployment conditions before making
HPC, desktop/Spark or edge acceptance claims.
