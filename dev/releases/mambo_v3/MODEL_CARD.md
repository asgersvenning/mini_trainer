# MAMBO V3

Unpublished release candidate: EfficientNetV2-S trained on global-lepi in September
2026. Predicts 12,632 species, 4,476 genera and 104 families, identified by GBIF taxon
IDs. Native PyTorch and standard floating-point ONNX artifacts share this vocabulary.
No quantized model is included. Embeddings have 1,280 dimensions and unit length.

Use the release README for installation, input/output formats and configuration.
Global is the default. Region presets and custom class lists constrain eligible
species; they are permissive occurrence filters, not native-range maps. Taxonomic
ranks are predicted independently. Optional TTA uses `rotation30_pad25_3`.

## Intended use and evidence

Local moth/butterfly image classification and downstream integration. Both
Flemming monitoring crops and the original global-lepi test split have completed
V2/V3, backend and TTA comparisons. Their different domains produce different TTA
responses; neither establishes accuracy for every deployment. Regional vocabulary
and confidence thresholds affect the results. Consult the README's figures and
linked evidence for macro metrics, acceptance coverage, support truncation and
calibration policy. Laptop and B200 timings have distinct environment/workload
boundaries and do not establish universal hardware throughput.

The Python adapter supports CPU and NVIDIA CUDA via separately installed runtimes.
ONNX offers a path to browser and other native-runtime integrations; those require
matching preprocessing and are not automatically qualified by Python execution.
No complete Windows/macOS/edge-device compatibility claim is made.

## Training and artifact identity

The checksum-verified training console reports the best model at epoch **30**.
`MODEL_PROVENANCE.toml` identifies the checkpoint, configuration, epoch summary and
training log with immutable hashes and public source URLs. The retained materials
do not identify the exact training Git revision or retain the starting checkpoint
hash. The preparation source initializes a torchvision DEFAULT EfficientNetV2-S
backbone (ImageNet-1K) and a new hierarchical head with seed 42; this is a source-based
reconstruction rather than a verified identity for the original starting file. The recorded September 11 checkout is packaging provenance,
not a claimed training revision. The trained checkpoint itself is identified and
can be loaded without retraining or downloading an initialization model.

`release.json` covers the graph/external-weight files, presets, preprocessing,
vocabulary and documentation. `PRESET_DEFINITIONS.toml` and `PRESET_UPDATES.toml`
record list construction. Read `NOTICES.md` for code, model and source-data boundaries.

## Publication status

The candidate weights are prepared under **CC BY-NC-SA 4.0**: attribution,
non-commercial use and share-alike terms for distributed adaptations. See
`MODEL_LICENSE.txt` and `NOTICES.md` for the terms and upstream attribution.
The adapter code remains MIT-licensed. This candidate has not been published.
