# Releases: trunk, heads and a Hugging Face folder

`mt_release` turns a checkpoint into a folder that runs without mini_trainer, checks it against
mini_trainer on real images, and prepares it for Hugging Face. Nothing is uploaded: the last step
prints the command for a person to run. `mt_export` ([ONNX export](onnx.md)) remains the generic,
raw-forward graph export; `mt_release` is the deployment package built on the same loaders.

```bash
mt_release export --weights last.pt --output release/ --ranks species,genus,family \
    --head-name main --images test-images.txt          # writes the release, then verifies it
mt_release package-hf --release release/ --repo-id user/model --card card.json
```

`--images` is a folder, a text file of paths or a JSON list (or `{"path": [...]}`); 300 are sampled
(`--limit`). Without it the release is written but not verified, and `package-hf` says so.

## What a release contains

safetensors is the release format (PyTorch, Hugging Face, fine-tuning); ONNX is the deployment format
(ONNX Runtime on CPU or GPU, and the entry point to TensorRT, OpenVINO and Core ML converters).

The model is cut at the classifier's input. The **trunk** is everything before it; the **head** is the
classifier, hidden layer included, because `mt_train --freeze-backbone` trains a new hidden layer on a
frozen trunk.

| file | content |
|---|---|
| `trunk.onnx`, `trunk_fp16.onnx` | `images` (uint8 `[N, 3, read, read]`) -> `features` (classifier input) and `embedding` (the default head's normalised embedding). Resampling to the model size and normalisation are in the graph. fp16: half-precision backbone, fp32 head layers. One file when under 2 GB. |
| `trunk.safetensors` | trunk weights under the backbone's own names (timm backbones load into `timm.create_model(name, num_classes=0)` strictly) |
| `heads/<name>.safetensors` | `direction` (unit prototypes), `magnitude` (g), `bias` (b), `hidden.*`, `parent.<k>` (each node's parent at rank k + 1); metadata: ranks, sizes, `magnitude_after_angle` |
| `heads/<name>.json` | rank names and labels in output order |
| `model.safetensors` | trunk and default head in one file (`trunk.*`, `head.*`), tensors identical to the separate files |
| `model.onnx` (+ `.onnx.data`) | trunk and default head in one graph -> `embedding`, `logits_<rank>` (mini_trainer's eval outputs) |
| `config.json` | preprocessing, trunk (backbone, sha256 of its weights), heads, every file's size and sha256 |
| `verification.json` | the parity report below |
| `predict.py`, `README.md` | written by `package-hf` |

`predict.py` is `mini_trainer/release/standalone.py`, copied: numpy, Pillow, onnxruntime and
safetensors only. It reads images exactly as the evaluation loader does (`read_square_crop` at the
centre), scores a head as mini_trainer does
(`sqrt(D-2)*asin(clamp(g*cos)) + b`, or `g*sqrt(D-2)*asin(clamp(cos)) + b` with `magnitude_after_angle`;
coarser ranks are logsumexp over children), restricts a head to a class list (`subset`: rows, with
coarser ranks re-indexed like mini_trainer's class masks) and can rebuild a timm trunk from
`trunk.safetensors` (`torch_trunk`).

Supported: normalised (cosine) `Classifier` and leaf-aggregated `HierarchicalClassifier` heads,
square-crop preprocessing (`resize_method="crop"`). Other heads and preprocessing are refused.

## Several heads on one trunk

```bash
mt_release add-head --weights frozen-trunk-head.pt --release release/ --name moths --ranks species,genus,family
```

The head is added only if the checkpoint's trunk weights hash to the release's trunk and its
preprocessing matches; a fine-tuned trunk is refused and needs its own release.

## Verification

`verify` (run by `export` and `add-head`, or alone: `mt_release verify --weights last.pt --release
release/ --images ...`) loads the checkpoint with mini_trainer, computes features, the embedding (via
`EmbeddingContext`) and every rank's logits, and compares each artifact run through `predict.py`'s own
code: reader pixels, `trunk.onnx` and `trunk_fp16.onnx` with the head file, `model.onnx`, and
`trunk.safetensors` through timm. It records per-rank top-1 agreement, maximum and mean logit
differences, embedding cosine, file sizes and CPU latency. Gates: fp32 artifacts at least 99 %
top-1 agreement at every rank and embedding cosine 0.9999; fp16 95 % and 0.999. This is agreement
with mini_trainer, not accuracy.

## Hugging Face

`package-hf` copies `predict.py`, writes `README.md` from the release, `verification.json` and a card
file, refreshes the hashes, runs `predict.py` in an isolated interpreter (it must not import torch,
timm or mini_trainer), and prints `hf upload <repo-id> <folder> . --repo-type model`. The card file is
JSON written by the model's owner:

```json
{"title": "...", "summary": "...", "license": "other", "license_name": "...", "license_link": "LICENSE.md",
 "license_files": ["path/to/LICENSE.md"], "license_section": "...", "base_model": "...", "tags": ["..."],
 "training": "...", "results": [{"claim": "...", "source": "where the number is recorded"}],
 "limitations": ["..."], "citation": "@misc{...}"}
```

Without `license` (or with failed verification, missing files, or a `predict.py` that does not run)
the folder is reported as not ready. Licences are never inferred: a trunk derived from licensed
weights (for example DINOv3) and training images with mixed licences need the owner's decision.
