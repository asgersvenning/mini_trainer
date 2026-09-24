# MAMBO deployment — release candidate

Run a local model bundle with ONNX or PyTorch, on CPU or NVIDIA CUDA. Inference
requires no network access or writable model directory. This candidate has not
been publicly released. Keep each ONNX graph beside its `model.onnx.data` file.

## Quick start

Start with ONNX/CPU for the smallest installation: it needs no training package.
Install the supplied wheel in your environment, then reuse one predictor across calls.

```sh
pip install './mambo_deploy-0.3.0-py3-none-any.whl[onnx]'
```

```python
from mambo_deploy import Predictor

predictor = Predictor(bundle="/path/to/mambo-bundle", backend="onnx", device="cpu", model="europe")
result = predictor.predict(["moth.jpg"])
print(result[0].label)       # species, genus, family IDs
print(result[0].confidence)  # confidence at each rank
```

For ONNX/CUDA, install `onnxruntime-gpu` instead of `onnxruntime`, with matching
CUDA/cuDNN libraries, and select `device="cuda:0"`. For PyTorch, install the matching
`mini_trainer` wheel and CPU/CUDA PyTorch build, then select `backend="torch"` and
an explicit device. Requested but unavailable CUDA raises an error; individual
ONNX operators may still execute on CPU. CPU and CUDA are the supported device
choices; other OS/accelerator combinations remain unqualified.

## Choose the configuration that matters

| Setting | Starting point | When to change it |
|---|---|---|
| `model` / `class_list` | Set the region explicitly; default is `europe` | Match your sampling location. Use `full` when geography is unknown, or a custom list for your project's eligible species. This changes predictions and confidence. |
| `backend`, `device` | ONNX/CPU for portable integration | Use CUDA for throughput. On this laptop, ONNX was faster on CPU; PyTorch scaled better at GPU batch 32. Choose by dependencies and measurements on your hardware. |
| `tta` | Off | Enable `tta=True` when improved quality justifies three model passes. Keep the recommended recipe unless you validate an alternative on your own data. |
| `batch_size` | `8` | For GPU bulk processing, try 8 then 32; reduce for memory limits or interactive requests. Larger batches do not guarantee higher throughput. |
| `threads`, `preprocess_workers` | `threads=2`; preparation workers follow it | Tune under the real application's CPU budget. Preparation workers handle decoding/transforms; `threads` also controls ONNX runtime threads, but does **not** set PyTorch model threads. Avoid multiplying workers across competing processes. |
| `precision` | `"auto"` | Usually leave it alone. Use `"fp32"` to investigate runtime/numerical issues. BF16 is a native CUDA option requiring hardware support, not an established improvement over the default. |
| Embeddings / `topk` | Predictions only; `topk=1` | Request embeddings for similarity/search or downstream features; request more candidates with `predict(images, topk=k)`. Neither improves the classifier itself. |

`precision="auto"` means FP32 on CPU, FP16 backbone with FP32 head for native
CUDA, and TF32 execution of the standard floating ONNX graph on CUDA. It does not
select quantized weights or an FP16 ONNX export. Results record the resolved
precision, preset, class-list hash and TTA recipe; retain these with the bundle
version when comparing runs.

### Geographic scope

Use `predictor.available_presets()` and the bundle's `PRESETS.md` to choose a list;
the [preset catalogue](../docs/model-presets.md) documents exact scope and construction.
Presets cover species that **can occur** in a region, including introduced species;
they are neither native-distribution maps nor exhaustive checklists.

`europe` and `north_europe` preserve legacy lists. The `_v3` alternatives use updated
occurrence requirements and broader eligibility; newer does not necessarily mean
more accurate. Legacy `north_europe` performed better on Flemming. Choose it for
comparable northern-European use, not as a universal default for other locations.
A custom `class_list=["GBIF_SPECIES_ID", ...]` or UTF-8 list file overrides the preset.
Unknown IDs and empty lists fail; duplicates are removed and model ordering retained.

### Optional test-time augmentation

`tta=True` or bare `--tta` selects `rotation30_pad25_3`: original, −30° and +30°
views, with 25% edge padding on each rotated view. Rotation expands the canvas;
ordinary model preprocessing follows. Pin that name explicitly for reproducibility.
TTA remains off when omitted. `padded_scale` preserves the previous recipe;
`wide_rotation_mixed_padding_5` is an optional higher-cost alternative with no
consistent family-level advantage. See the [TTA reference](../docs/mambo-tta.md)
for custom transforms and aggregation details.

## Inputs, outputs and acceptance

Supply paths, PIL images, or CHW/BCHW arrays/tensors: uint8, or floats in [0,1].
Transpose HWC arrays explicitly. Supply original pixels, **not normalized model
inputs**. The runtime converts to RGB, discards alpha and ignores EXIF orientation;
apply any required orientation correction before passing a PIL image or array.
Keep the supplied preprocessing recipe unchanged.

Results are on CPU. `label` contains taxon IDs in species/genus/family order;
`topk` ranks each level independently, so a returned tuple need not be an ancestral
path. `indices` refer to the filtered vocabulary; `global_indices` to the full one.
`predict_with_embeddings(images)` returns `(result, vectors)`, with float32
`[N,1280]` unit-length vectors; the ONNX bundle must include the embedding graph.
With TTA, embeddings are the normalized mean across views.

`batch_size` bounds model calls, **not total request memory**: outputs and logits
accumulate for the whole request. Submit large collections in bounded chunks.
Calls on one predictor are serialized; increasing caller threads alone will not
parallelize its inference.

Confidence is conditional on the selected vocabulary, not a guarantee of correctness.
If your workflow can reject uncertain predictions, choose thresholds on representative
labelled data and report acceptance coverage alongside quality. Recalibrate when
changing the preset, TTA or pipeline. The study thresholds below are not universal
production defaults. In Python, apply your acceptance rule to `result.confidence`.
CLI `--threshold` defaults to zero and marks acceptance in `mini_metric.csv`; it
uses one scalar for all ranks and does not remove predictions from JSON output.

## Command line and migration

```sh
mambo_predict -i moth.jpg --bundle /path/to/mambo-bundle --backend onnx --device cpu -M europe --tta -o . --name results
```

Outputs go to a new `results/` directory: `predictions.json`, `mini_metric.csv`, and
`embeddings.npy` when `--embeddings` is requested. Directory input is recursive.
The main controls above have corresponding CLI flags; use `mambo_predict --help`.

Existing callers can use `mini_trainer.deploy.Predictor` with both wheels installed.
It preserves native/CUDA defaults, callable prediction and `class_mask` (`-1` resets
it), with native result containers/device tensors. The portable API above defaults
to ONNX/CPU. Pass `bundle=` or set `MAMBO_BUNDLE`; downloads are no longer implicit.
Do not use `weights=` as a model-selection control: overrides must match the pinned
release checkpoint. Legacy weights and already-preprocessed inputs need migration;
embedding dimensions may differ from V2.

## Release comparison

These results help choose TTA and runtime; they do not establish accuracy in every
region. All models use legacy northern Europe on the same 52,788 Flemming reporting
images, including out-of-vocabulary truth. `mini_metrics` selects calibrated
thresholds per pipeline/rank on 5,852 separate images. Recipe exploration used
Flemming too, so this is descriptive evidence, not independent validation.

The quality figure compares **unthresholded and calibrated predictions**, with
full-support and support >5 macro metrics alongside acceptance coverage. TTA uses
`rotation30_pad25_3`; its quality gain comes at the throughput cost shown below.

![Species, genus and family quality: full and truncated support, both confidence settings, and coverage](../docs/assets/mambo-promoted-quality.svg)

Support >5 requires more than five truth instances and accepted predictions in
every compared pipeline. Truth classes outside that average account for
15.30% / 0.42% / 0.01% of images at species/genus/family without thresholds,
and 18.95% / 11.21% / 0.03% after calibration. No evaluation rows are discarded;
the averaging class sets differ between confidence settings. Full-support metrics
retain rare and predicted-only classes, which can change model rankings.

Measured **images/second**, end to end, on an i7-12800H / RTX 3080 Ti Laptop with
four preparation/runtime threads. V3 uses automatic precision; compare on your own
hardware before choosing a batch size. V2 and single-view V3 reuse earlier runs.

![CPU and GPU throughput by batch size, including the new TTA default](../docs/assets/mambo-promoted-speed.svg)

On this laptop, ONNX is faster on CPU; native PyTorch benefits more from larger
GPU batches. TTA improves quality but reduces throughput, so enable it according
to your accuracy and processing-budget requirements.

The [complete evidence reference](../docs/mambo-deployment-evidence.md) retains
exact metric tables, calibrated thresholds, timing ranges and limitations.
In-domain UCloud evaluation remains outstanding.
