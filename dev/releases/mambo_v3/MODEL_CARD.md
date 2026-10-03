---
license: cc-by-nc-sa-4.0
library_name: mambo-deploy
model_name: Nemo
pipeline_tag: image-classification
tags:
  - biology
  - lepidoptera
  - moths
  - butterflies
  - pytorch
  - onnx
---

# Nemo

Nemo (MAMBO_v3) identifies adult moths and butterflies in photographs, predicting
{{vocabulary}}. Predictions use GBIF taxon IDs.

[Try Nemo](https://huggingface.co/spaces/asgersvenning/MAMBO-v3) ·
[Installation and API](https://github.com/asgersvenning/mini_trainer/blob/models/mambo-v3/v0.3.1/deployment/README.md) ·
[Source code](https://github.com/asgersvenning/mini_trainer)

Install with `pip install 'mambo-v3[onnx,hub]==0.3.1'`:

```python
from mambo_deploy import Predictor

model = Predictor.from_pretrained("asgersvenning/MAMBO-v3", backend="onnx")
prediction = model.predict("moth.jpg")[0]
print(prediction.label, prediction.confidence)
```

## Performance

The comparisons cover three image domains:

1. **GBIF training domain:** held-out test images from the same collection used
   to train Nemo (global scope; left panel).
2. **Automated monitoring:** Danish AMI camera-light-trap image crops,
   expert-reviewed by Flemming Helsing (northern European scope; right panel).
3. **iNaturalist:** recent research-grade images from its own platform, comparing
   Nemo, Nemo + TTA and Meghan with iNaturalist’s model (below).

![Nemo and Meghan on GBIF test images and automated monitoring images](performance/nemo-card-quality.svg)

Species-level scores after confidence filtering. Solid bars show full support;
pale bars show shared support >5. Acceptance rates appear below each model.

**Suggested species-confidence thresholds for Nemo ONNX (without / with TTA):**
**0.53 / 0.35** for general photographs (global scope);
**0.82 / 0.75** for image crops of single individuals from camera light traps
(e.g. AMI; `north_europe`).

### Recent research-grade iNaturalist images

{{performance}}

## Variants and details

- **ONNX:** CPU inference without PyTorch; the default Python backend.
- **PyTorch:** native CPU/CUDA inference through `mini_trainer`, or the same portable API.
- **Scope:** global by default; regional presets and custom species lists restrict predictions.
  `model="europe"` selects geography, not the model generation.
- **Embeddings:** {{embedding_dim}} dimensions for downstream applications.

Nemo uses EfficientNetV2-S, trained for {{epochs}} epochs on {{training_images}}
GBIF-sourced training images in the global-lepi collection.
[Training configuration]({{training_config}}) ·
[Provenance](MODEL_PROVENANCE.toml) ·
[Migration from Meghan (MAMBO_v2)](https://github.com/asgersvenning/mini_trainer/blob/models/mambo-v3/v0.3.1/docs/mambo-integration.md#moving-from-v2)

## License

Model weights: **[CC BY-NC-SA 4.0](MODEL_LICENSE.txt)** (attribution, non-commercial,
share-alike). Code: **MIT**. See [notices and attribution](NOTICES.md).
