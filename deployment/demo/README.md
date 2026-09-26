---
title: MAMBO V3
emoji: 🦋
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: "6.28.0"
python_version: "3.13"
app_file: app.py
suggested_hardware: cpu-basic
tags:
  - image-classification
  - biodiversity
  - onnx
license: mit
---

# MAMBO V3 demonstration

One image, configurable runtime, geographic/custom class list, TTA and hierarchical
predictions. The Space uses the release API on CPU and caches one predictor per backend
(two in total), reusing loaded runtimes when scope or TTA changes. It does not require GPU hosting. First predictions include loading.

Uploaded images are processed on the server. Gradio's temporary upload cache is
cleaned every five minutes for files older than five minutes; the application does
not save images, collect feedback or use uploads for training. Do not upload
sensitive images. Prediction requests are serialized and the queue is bounded.

The application code is MIT. Model weights are **CC BY-NC-SA 4.0**. See the
[release documentation](https://github.com/asgersvenning/mini_trainer/blob/MAMBO_v3/deployment/README.md)
for integration, comparisons and limitations. Readable names are displayed when
provided in the accompanying `taxon-names.json`; GBIF IDs remain authoritative.

The release workflow stages this directory with pinned dependency requirements
and optional taxon names. It uploads only that staged directory, not the repository.
