---
license: cc-by-nc-sa-4.0
library_name: onnx
pipeline_tag: image-classification
tags:
  - biology
  - lepidoptera
  - moths
  - butterflies
  - pytorch
  - onnx
  - hierarchical-classification
---

# MAMBO V3

EfficientNetV2-S trained on global-lepi in September
2026. Predicts 12,632 species, 4,476 genera and 104 families, identified by GBIF taxon
IDs. Native PyTorch and standard floating-point ONNX artifacts share this vocabulary.
No quantized model is included. Embeddings have 1,280 dimensions and unit length.

[Try one image](https://huggingface.co/spaces/asgersvenning/MAMBO-v3) ·
[Python package](https://pypi.org/project/mambo-v3/) ·
[Integration and comparison figures](https://github.com/asgersvenning/mini_trainer/blob/MAMBO_v3/deployment/README.md)

```python
from mambo_deploy import Predictor
result = Predictor().predict("moth.jpg")
print(result[0].label, result[0].confidence)
```

Install with `uv pip install 'mambo-v3[onnx]==0.3.0'`. The package downloads verified
weights from ERDA automatically. For a Hub snapshot, use its `bundle/` directory
with `Predictor(bundle="/path/to/snapshot/bundle")`; this is not a Transformers
`from_pretrained` model. `CITATION.cff` identifies the release citation.

Use the release README for installation, input/output formats and configuration.
Global is the default. Region presets and custom class lists constrain eligible
species; they are permissive occurrence filters, not native-range maps. Taxonomic
ranks are predicted independently. Optional TTA uses `rotation30_pad25_3`.

## Intended use and evidence

Local moth/butterfly image classification and downstream integration. This is a
closed-vocabulary classifier for images of individual animals, not an animal
detector or a validated unknown-species rejection system. Both
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

## License

The model weights use **CC BY-NC-SA 4.0**: attribution,
non-commercial use and share-alike terms for distributed adaptations. See
`MODEL_LICENSE.txt` and `NOTICES.md` for the terms and upstream attribution.
The adapter code remains MIT-licensed.
