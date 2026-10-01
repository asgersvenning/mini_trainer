# Nemo integration details

Start with the [deployment quickstart](../deployment/README.md). This page covers
runtime choices, restricted environments and optional tuning; these are not
additional steps for the default ONNX/CPU integration.

## Runtime installation

Use an activated Python 3.12+ environment. The commands target the prepared public release; reviewers can substitute the
supplied wheels before publication. Choose one runtime installation:

| Environment | Installation | Predictor options / CLI |
|---|---|---|
| CPU, without PyTorch | `uv pip install 'mambo-v3[onnx]==0.3.1'` | Defaults: `backend="onnx", device="cpu"` / `--backend onnx --device cpu` |
| NVIDIA GPU, without the training package | `uv pip install 'mambo-v3[onnx-cuda]==0.3.1'` | `backend="onnx", device="cuda:0"` / `--backend onnx --device cuda:0` |
| PyTorch CPU or NVIDIA GPU | `uv pip install --torch-backend=auto 'mambo-v3[torch]==0.3.1'` | `backend="torch", device="cpu"` or `device="cuda:0"` / `--backend torch --device cpu` or `--device cuda:0` |

For an environment with ONNX Runtime already provisioned, install the base
`mambo-v3==0.3.1` without extras. Do not install CPU and GPU ONNX Runtime
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

## Loading and metadata

`mini_trainer.deploy.Predictor` owns native PyTorch inference and loads eagerly.
Install `mt-trainer>=0.3.1`; no deployment package is needed. It defaults to CUDA
and `europe`, preserving Meghan's (MAMBO_v2) calling conventions while using
Nemo (MAMBO_v3) weights. Supported local checkpoints remain usable through
`model="weights.pt"` or `weights=state_dict`; `weight_dir` controls the default
checkpoint cache. Optional backbone dependencies are still needed for older models.

`mambo_deploy.Predictor` defaults to ONNX/CPU and global scope. It delegates Torch
execution to the native predictor. Construction reads metadata; `load()` loads the
runtime without an image and returns the predictor. Repeated calls reuse the model.
Use `load(embeddings=True)` to also prepare the separate ONNX embedding graph.
Loading validates files and runtime availability; it does not perform inference or
promise that every kernel has been warmed up.

```python
from mambo_deploy import Predictor

predictor = Predictor(backend="torch", device="cuda", model="europe").load(embeddings=True)
print(predictor.input_size, predictor.embedding_dim)
print(predictor.metadata)
result, vectors = predictor.predict_with_embeddings("moth.jpg")
```

Both predictors expose `metadata`, `input_size` (square input side), `classes`
(ordered labels by rank), `cls2idx` (rank-keyed full-vocabulary mappings),
`embedding_dim`, and `preprocessing` as defensive snapshots. `class_list` identifies
the active species selection; prediction mappings refer to the selected vocabulary.
Native callers can additionally access `.model`, `.preproc`, `.reader`, `.weights`
and `.source`. No dummy image or private-attribute access is needed for startup.
The portable API returns CPU arrays on either backend; the native API retains
Torch prediction containers and device tensors.

### Hugging Face Hub

Install the `hub` extra alongside the selected runtime, for example
`mambo-v3[onnx,hub]==0.3.1`. Load directly with:

```python
predictor = Predictor.from_pretrained(
    "asgersvenning/MAMBO-v3", revision="<published commit SHA>",
    backend="onnx", embeddings=True,
)
```

`from_pretrained` accepts a local bundle or Hub snapshot too. Hub downloads use
`huggingface_hub`'s `cache_dir`, `token`, `force_download` and `local_files_only`
options. A branch resolves once to an immutable revision. Only the selected
backend's assets are retrieved; later embedding requests use that same revision.
Offline operation requires those files to be cached already. Bundle hashes are
validated after download. This is Hub-native loading, not Transformers `AutoModel`,
`pipeline`, or a hosted inference endpoint.

## Moving from V2

Meghan is the team's existing alias for MAMBO_v2; Nemo is the public name for
MAMBO_v3. Technical identifiers, package names and existing artifact URLs remain
unchanged. Package 0.3.1 restores the independent native API; the original 0.3.0
wrapper required the deployment package and defaulted to global scope.

Native callers retain Europe/CUDA defaults, local checkpoint inputs, callable
prediction and `class_mask` (`-1` resets it). Portable callers explicitly select
`europe` or `north_europe` for the legacy lists; `_v3` presets change eligibility.
Supply original pixels, not tensors normalized by a previous model's pipeline.
Match class identities using GBIF IDs rather than positions. Nemo embeddings have
width 1,280; regenerate Meghan similarity indexes and recalibrate confidence
thresholds. Compatible calls do not imply identical model predictions.

The portable `weights=` override must still match the bundle's checkpoint; use the
native predictor for arbitrary supported checkpoints. `model=` selects geography,
not a model generation. The portable `configure(model=..., class_list=..., tta=...)`
method changes selection/TTA without reloading the model. Finish active streams
before reconfiguring.

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
`mambo-v3==0.3.1` to preserve the adapter implementation too, and retain your
application's resolved runtime dependencies for reproducibility.

The Python namespace stays `mambo_deploy`. Do not install multiple model-generation
packages or the older `mambo-deploy` candidate in one environment; use separate
environments for comparisons. The `model=` argument selects geographic scope,
not a trained-model release. Result metadata identifies the model and selected
list. Explicit local bundles are an advanced override, not an automatic upgrade.

Model files are cached by their SHA-256 identity and reused across metadata-only
updates. Copies placed beside ONNX graphs keep bundles relocatable; there is no
runtime package installation or mutable remote "latest model" lookup.
