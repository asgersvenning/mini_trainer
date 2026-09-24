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

**Choose your geographic scope and runtime explicitly.** Start with ONNX/CPU for
simple integration, or use your existing PyTorch/CUDA environment. Keep
`precision="auto"` and the default TTA recipe; these are better starting points
across machines than copying benchmark-specific settings.

Leave TTA off for throughput, or enable `tta=True` when quality matters more.
Start with the default batch size and worker counts. Tune those only when speed
or memory becomes limiting, using representative inputs on the target machine;
the laptop's best batch size need not be yours. Request embeddings or extra
candidates only when your application needs them.

| Setting | Default | Role / main trade-off |
|---|---|---|
| `model` / `class_list` | `europe` / no override | Prediction scope: selects eligible species and changes confidence. |
| `backend`, `device` | `onnx`, `cpu` | Runtime dependencies, hardware compatibility and throughput. |
| `tta` | Off; `True` selects `rotation30_pad25_3` | Quality versus compute: the recommended recipe uses three views. [Recipe details](../docs/mambo-tta.md). |
| `batch_size` | `8` | Throughput and working memory: images per model call, not a total-request memory limit. |
| `threads` | `2` | CPU allocation: ONNX runtime threads and the default preparation-worker count; does not set PyTorch model threads. |
| `preprocess_workers` | Follows `threads` | CPU preparation concurrency: decoding and transforms can compete with other application work. |
| `precision` | `auto` | Compute speed and numerical precision: selects the backend/device's default mode. |
| Embeddings | Off | Additional output for similarity/search or downstream features. |
| `topk` | `1` | Number of returned candidates at each taxonomic rank. |

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
