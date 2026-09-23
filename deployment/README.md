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
For ONNX CUDA, install `onnxruntime-gpu` instead of the CPU ONNX Runtime package,
with its matching CUDA/cuDNN dependencies. Select `device="cuda:0"` explicitly.
Unavailable CUDA raises an error rather than silently changing to CPU-only
execution. ONNX CUDA may still place individual unsupported operators on CPU.

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
controls ONNX CPU threads; native callers configure PyTorch threads. The standalone
wheel defaults to ONNX/CPU; the training-wheel entry point retains native/CUDA
defaults, so explicit backend/device arguments are recommended in scripts.

## Qualification

Preset generation and artifact integrity have been checked. This increment adds
small real-image inference and installed-package checks, not full performance or
accuracy qualification. Full Flemming metrics, in-domain UCloud evaluation,
laptop GPU/CPU benchmarks, other operating systems and publication/license
review remain release work. Small ONNX numerical differences are expected.
