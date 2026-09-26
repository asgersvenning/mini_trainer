# MAMBO integration details

Start with the [deployment quickstart](../deployment/README.md). This page covers
runtime choices, restricted environments and optional tuning; these are not
additional steps for the default ONNX/CPU integration.

## Runtime installation

Use an activated Python 3.12+ environment. The commands target the prepared public release; reviewers can substitute the
supplied wheels before publication. Choose one runtime installation:

| Environment | Installation | Predictor options / CLI |
|---|---|---|
| CPU, without PyTorch | `uv pip install 'mambo-v3[onnx]==0.3.0'` | Defaults: `backend="onnx", device="cpu"` / `--backend onnx --device cpu` |
| NVIDIA GPU, without the training package | `uv pip install 'mambo-v3[onnx-cuda]==0.3.0'` | `backend="onnx", device="cuda:0"` / `--backend onnx --device cuda:0` |
| PyTorch CPU or NVIDIA GPU | `uv pip install --torch-backend=auto 'mambo-v3[torch]==0.3.0'` | `backend="torch", device="cpu"` or `device="cuda:0"` / `--backend torch --device cpu` or `--device cuda:0` |

For an environment with ONNX Runtime already provisioned, install the base
`mambo-v3==0.3.0` without extras. Do not install CPU and GPU ONNX Runtime
packages together. Use the application's dependency management to select and
record versions; inference never installs or replaces runtime packages. If you
use `uv run`, pass `--no-sync` to retain the installed environment.

ONNX removes the training-package dependency and provides the same Python and CLI
interface on CPU or CUDA. The adapter currently exposes only CPU and NVIDIA CUDA;
it does not automatically enable other ONNX execution providers. Availability of
a compatible runtime wheel and adequate memory still matters on edge hardware.
Linux measurements do not establish support for every OS or accelerator.

CUDA requires a compatible NVIDIA driver and runtime build. The `onnx-cuda` extra
requests CUDA/cuDNN dependencies; it cannot guarantee that a wheel includes kernels
for every GPU architecture. The B200 evidence uses ONNX Runtime 1.22.0, recorded in
[the measured environment](mambo-hpc-evidence.md), not a universal version pin.
Requested unavailable CUDA raises an error rather than silently switching the
whole model to CPU; ONNX may place individual operators on CPU.

ONNX/CUDA probes each graph once when first loaded. A GPU-kernel compatibility
failure triggers a checked retry with graph optimizations disabled and a warning.
If that also fails, inference stops with an error. Sessions are reused; inspect
`predictor.onnx_session_info` when diagnosing a runtime problem. This check does
not guarantee every batch-dependent execution path.

## Restricted and offline environments

Models are cached in `~/.cache/mambo` (or `$XDG_CACHE_HOME/mambo`). Set `MAMBO_CACHE`
to a writable persistent directory when the default home is unsuitable.
First use requires outbound access to the public ERDA model files; later calls
reuse verified assets.

For deployment without network access, provision dependencies beforehand and either:

- Run the intended backend and output mode on a connected machine, copy its model
  cache, set `MAMBO_CACHE` to that location and set `MAMBO_OFFLINE=1`. Include a call
  with embeddings if the application will request them; that uses another ONNX graph.
- Supply a complete release bundle and pass `Predictor(bundle="/path/to/bundle")`,
  `--bundle /path/to/bundle`, or set `MAMBO_BUNDLE`. Keep each ONNX graph beside its
  external `model.onnx.data` file. An explicit bundle is read locally.

The same model files serve CPU and CUDA for each backend. No training dataset or
metadata parquet is required. Prediction results and model caches are separate:
the API returns results to the application; the CLI writes to its chosen output
directory. Raw PyTorch checkpoints require the matching architecture/runtime;
use the release adapter rather than loading them as arbitrary models.

## Moving from V2

For existing callers, `mini_trainer.deploy.Predictor` preserves native result
containers/device tensors, native/CUDA defaults, callable prediction and
`class_mask` (`-1` resets it). The portable `mambo_deploy.Predictor` instead defaults
to ONNX/CPU and returns CPU results. The two entry points share release model assets.

Keep `europe` or `north_europe` for the legacy preset; `_v3` presets deliberately
change eligibility. Supply original pixels to the portable API, not tensors
normalized by an old preprocessing pipeline. Match class identities using GBIF
IDs rather than positions. V3 embeddings have width 1,280; recreate stored
embeddings if migrating a similarity index from V2. Confidence thresholds are
model/list specific.

`weights=` is not a V2/V3 selection switch: a native override must match the pinned
release checkpoint. Keep the V2 runtime/assets separately if you still need to run
V2. Preserving its calling conventions does not imply identical predictions or a
shared embedding space.

For an interactive application, call `predictor.configure(model="north_europe", tta=True)`
to change scope and TTA while reusing loaded models. Omitted settings stay unchanged;
`class_list=[...]` selects custom species, `model="full"` resets the scope, and
`tta=False` disables TTA. Finish any active stream before reconfiguring.

## Streaming controls

`predict_stream(paths)` yields ordered prediction batches; `embeddings=True` yields
`(prediction, vectors)` pairs. Consume each batch without retaining it to keep
output memory bounded. Use `contextlib.closing` when stopping early; shutdown waits
for filesystem calls already in progress.

Start with defaults, then change the resource relevant to your workload. These
options belong to `predict_stream`, not the constructor or CLI:

| Option | Default | When it matters |
|---|---|---|
| `read_workers` | `32` | Concurrent file reads, useful for storage latency. |
| `read_window` | `max(128, batch_size)` images | Maximum lookahead; follows larger batches automatically unless explicitly set. |
| `prepare_workers` | Predictor's `preprocess_workers` | Decoding and image preparation; shares CPU capacity with your application. |
| `prefetch_batches` | `2` | Prepared input buffer size; trades memory for overlap. |
| `encoded_budget` | `256 * 1024**2` bytes | Encoded image buffer budget; a larger single file fails explicitly. |

These budgets do not bound total process memory: model weights, decoded images,
prepared views and results also consume memory. `predict()` accumulates results
for the whole input collection; submit bounded requests there. The CLI streams
predictions and embeddings to disk and publishes the output directory only when
the complete run succeeds.

For diagnostics, `stats={}` collects queue/buffer and wait statistics;
`device_prefetch=False` disables device staging. These are not routine integration
settings. Calls using one predictor share serialized inference; adding caller
threads alone does not create concurrent model execution.

## Versioning and model identity

The distribution name `mambo-v3` identifies the model generation. Package updates
within that distribution retain the released V3 weights and existing preset
identities; changed weights belong to a new model-generation package. Pin
`mambo-v3==0.3.0` to preserve the adapter implementation too, and retain your
application's resolved runtime dependencies for reproducibility.

The Python namespace stays `mambo_deploy`. Do not install multiple model-generation
packages or the older `mambo-deploy` candidate in one environment; use separate
environments for comparisons. The `model=` argument selects geographic scope,
not a trained-model release. Result metadata identifies the model and selected
list. Explicit local bundles are an advanced override, not an automatic upgrade.

Model files are cached by their SHA-256 identity and reused across metadata-only
updates. Copies placed beside ONNX graphs keep bundles relocatable; there is no
runtime package installation or mutable remote "latest model" lookup.
