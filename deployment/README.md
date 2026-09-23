# MAMBO deployment — release candidate

Use a local model bundle with native PyTorch or standard ONNX. Inference needs no
network, taxonomy service, administrator permissions or writable model directory.
The runtime is separate from the model files; keep each ONNX graph beside its
`model.onnx.data` file. This candidate has not been publicly released.

## Quick start

Install the supplied `mambo_deploy` wheel with the `onnx` extra for CPU inference.
For example, in a virtual environment:

```sh
pip install './mambo_deploy-0.3.0-py3-none-any.whl[onnx]'
```

```python
from mambo_deploy import Predictor

predictor = Predictor(bundle="/path/to/mambo-bundle", backend="onnx", device="cpu", model="europe")
result = predictor.predict("moth.jpg")
print(result[0].label)       # species, genus, family IDs
print(result[0].confidence)  # conditional probabilities for those ranks
result, embeddings = predictor.predict_with_embeddings(["moth.jpg"])
```

Choose a preset using `model`, inspect `predictor.available_presets()`, or replace
it with `class_list=["GBIF_SPECIES_ID", ...]` (also accepts a UTF-8 filename).
Unknown IDs and empty lists are errors; duplicates are removed and model ordering
is preserved. Filtering happens before ranking and hierarchy normalization.
See PRESETS.md in the bundle for short geographic scopes and provisional cutoffs.
Use `europe_v3` or `north_europe_v3` for updated lists requiring at least 3 regional
and 25 global records. `europe` (the default) and `north_europe` retain their legacy
membership. The updated versions use the same explicit geographic filters.
Presets retain species with qualifying occurrence records; they are practical
prediction filters, not maps of native distributions or exhaustive checklists.

Inputs are paths, PIL images, CHW/BCHW NumPy arrays or tensors. Arrays must be
uint8 or floats in [0,1]; transpose HWC arrays explicitly. Images are decoded RGB
without EXIF rotation, alpha is discarded, then the recorded campaign resize,
crop and normalization recipe is applied. Both backends use the same CPU image
preparation. `batch_size=8` bounds model batches; returned results remain in memory.

## PyTorch and GPU

For native inference, install the matching `mini_trainer` wheel with your chosen
PyTorch/CUDA build and use `backend="torch"`. The checkpoint is loaded with
`weights_only=True`; the architecture is constructed without pretrained downloads.
CUDA PyTorch defaults to FP16 backbone autocast with an FP32 classifier and
FP32 outputs. Set `precision="fp32"` to disable autocast, or
`precision="bf16"` on CUDA devices with native BF16 support. Native inference
respects the caller's PyTorch TF32 backend flags; reference benchmarks disable them.
For ONNX CUDA, install `onnxruntime-gpu` instead of the CPU ONNX Runtime package,
with its matching CUDA/cuDNN dependencies. Select `device="cuda:0"` explicitly.
Unavailable CUDA raises an error rather than silently changing to CPU-only
execution. ONNX CUDA may still place individual unsupported operators on CPU. Its automatic
precision enables TF32 using the standard FP32 graph; `precision="fp32"` disables
TF32. ONNX FP16/BF16 is not selected by PyTorch autocast. CPU always uses FP32.
The resolved choice is available as `predictor.effective_precision` and in result
metadata; the CLI exposes the same `--precision` option.

Portable results use NumPy arrays; embeddings are float32 `[images, 1280]` CPU
arrays from the normalized preclassification stage. Prediction-only ONNX uses
the original graph; requesting embeddings selects the existing embedding graph.
Each image uses one backbone pass. `topk` ranks each hierarchy level independently;
a tuple is not necessarily an ancestral path. `indices` refer to the filtered
rank vocabulary; `global_indices` refer to the full model vocabulary.

## Existing MAMBO callers

The matching training wheel restores `mini_trainer.deploy.Predictor`, with native
PyTorch and CUDA defaults, Europe as the default preset, callable/predict methods,
`class_mask` (including `-1` reset) and `(predictions, embeddings)` output. Install
the deployment wheel too and set `MAMBO_BUNDLE`, or pass `bundle=` explicitly.
Native compatibility results use the existing hierarchical prediction container
and device tensors. The portable API defaults to ONNX/CPU; choose it for new code.

Migration limits: inference no longer implicitly downloads a model; pass a local
bundle. Local `weights=` overrides must match this release checkpoint; arbitrary
legacy BioCLIP weights/state dictionaries are not supported by this release adapter.
Legacy pins remain the way to run those models. Embedding dimensions change with
the backbone. Legacy top-k serialization defects are not a portable API guarantee.
New portable preprocessing always applies the documented recipe to array inputs;
old callers that supplied already-resized or preprocessed tensors should review it.

## Command line

```sh
mambo_predict -i moth.jpg --bundle /path/to/mambo-bundle --backend onnx --device cpu -M europe -o . --name results
```

Directory input recursively discovers images in sorted order. Outputs are
`predictions.json` and `mini_metric.csv`; `--embeddings` also writes `embeddings.npy`.
Use `--class-list`, `--batch-size`, `--topk` and `--threads` as needed. `--threads`
bounds parallel image preparation and controls ONNX CPU threads; native callers
configure PyTorch model threads separately. Use `--threads 1` for serial preparation. The standalone
wheel defaults to ONNX/CPU; the training-wheel entry point retains native/CUDA
defaults, so explicit backend/device arguments are recommended in scripts.

## Optional test-time augmentation

TTA runs as an outer process: transform the decoded image, use the normal
preprocessing and backend, then average species logits before preset filtering
and hierarchical reduction. It is **off by default**.

```python
predictor = Predictor(bundle, backend="onnx", model="north_europe", tta="d4")
result = predictor.predict(images)
```

Whole-image profiles are `hflip` (2 views), `d4` (8 rotations/reflections),
and `light_noise` (original + two seeded 1% salt-and-pepper views). Experimental
`five_crop` and `ten_crop` profiles are also available; these can remove diagnostic
parts of a specimen and performed worse in the local subset comparison. Each model call stays within `batch_size`; images
are decoded once per batch. More views cost more inference and preparation.
Embeddings are the normalized mean of the view embeddings, not the single-view
representation. TTA quality qualification and exact semantics are in the
[TTA guide](../docs/mambo-tta.md).

Custom policies use the same outer layer, independently of backend or preset:

```python
from mambo_deploy import TTA, SaltAndPepper, View

policy = TTA((View(), View(quarter_turns=1), SaltAndPepper(seed=7)), name="orientation-noise")
predictor = Predictor(bundle, backend="torch", tta=policy)
```

A policy can also contain your own callables: each receives a separate decoded
uint8 CHW image and returns an image accepted by the normal preprocessing API.
The CLI supports the named profiles through `--tta`.

## Qualification

The release comparisons below use **TTA off**. Augmented results are reported
separately in the [TTA guide](../docs/mambo-tta.md).

In the strict FP32 reference evaluation on all 58,640 Flemming images, PyTorch and
ONNX return identical top-1 species,
genus and family labels for the full list and both legacy/updated European presets.
Northern Europe reaches **71.24% macro species accuracy** (v2: **68.52%**) and
**70.79% micro species accuracy overall** (**82.04%** on images
whose true species is in the list), versus **68.71% overall for MAMBO_v2**. Updated
northern Europe reaches **70.32% micro accuracy**. All predictive metrics use pinned `mini_metrics`, with threshold 0 and no
optimization. Unknown species remain in the overall result.
CPU/GPU and prediction/embedding variants agree on a fixed 256-image subset;
this checks prediction consistency, not downstream embedding usefulness.

On an i7-12800H / RTX 3080 Ti Laptop GPU, with four CPU threads and the legacy
northern-Europe preset, the updated defaults deliver:

| Runtime | CPU, batch 1 | GPU, batch 1 | GPU, batch 8 | GPU, batch 32 |
|---|---:|---:|---:|---:|
| MAMBO v2 (CPU input adapter) | 1.3 | 45.2 | 44.8 | 83.5 |
| V3 ONNX auto | 10.1 | 46.7 | 114.1 | 111.3 |
| V3 PyTorch auto | 5.7 | 29.8 | 126.7 | 136.2 |

All speeds are **images per second**, including decoding, preparation and result
handling. GPU batch-32 throughput improves by **2.62× ONNX / 3.06× PyTorch** over
the original v3 FP32 pipeline. CPU gains are not uniform. First prediction including
loading takes about 0.37 s CPU / 1.46 s GPU for ONNX, versus 42–43 s for PyTorch;
reuse a loaded predictor. ONNX uses less host memory and suits lightweight
integrations; native PyTorch suits existing callers and persistent GPU workers.

Both automatic GPU variants were evaluated on all 58,640 Flemming images. Northern-
Europe macro species accuracy is **71.25% PyTorch / 71.24% ONNX**; macro-F1 is
**0.2543** for both. The largest macro-accuracy change from FP32 across five presets,
three ranks and both truth populations is under 0.094 percentage points. BF16 has
4,096-image qualification only. Small cross-precision label differences are expected.

The unthresholded, all-truth **species** comparison is:

| Preset | V2 macro accuracy | V3 PyTorch / ONNX | V2 macro-F1 | V3 PyTorch / ONNX |
|---|---:|---:|---:|---:|
| Northern Europe | 68.52% | 71.25% / 71.24% | 0.2575 | 0.2543 / 0.2543 |
| Europe | 66.04% | 69.05% / 69.06% | 0.2000 | 0.1997 / 0.1997 |
| Global | 57.21% | 58.01% / 58.03% | 0.0899 | 0.0973 / 0.0973 |

The [frequency curves](../docs/mambo-frequency-comparison.md) compare both training
metadata counts and Flemming image counts. The [loading study](../docs/mambo-loading-scaling.md)
separates preparation from prepared-input inference: the remaining plateau is partly
loading/scheduling, not solely model throughput. `preprocess_workers` (CLI:
`--preprocess-workers`) controls preparation separately from ONNX runtime `threads`;
it defaults to the latter. Tune workers with batch size and CPU limits rather than
assuming more threads always help.

The [accelerated comparison and charts](../docs/mambo-accelerated-deployment.md)
cover speed, memory, all/known-truth metrics and updated European lists. The
[original v2/v3 comparison](../docs/mambo-release-comparison.md) preserves the FP32
reference and explains the metric definitions. V3 improves northern-Europe macro
accuracy over v2, but v2 retains slightly higher macro-F1 there and higher global
micro species accuracy. Choose presets for the deployment region; see the
[preset catalogue](../docs/model-presets.md).

The [evaluation workflow](../dev/releases/mambo_v3/evaluation.md) provides the
reproduction commands and UCloud handoff. In-domain evaluation, other operating
systems and publication/license review remain open. Small ONNX numerical
differences are expected even where top-1 labels agree.
