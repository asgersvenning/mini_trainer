# Browser inference in the prototype explorer

The explorer can run an embedding-enabled ONNX bundle locally using ONNX Runtime
Web 1.24.3 (WASM CPU). User images stay in the browser. The static server serves
model and application files; it performs no inference.

The first version accepts one opaque RGB JPEG or PNG at a time. Load the model,
choose an image, and inspect the five highest-scoring classes at each hierarchy
level. Species buttons select the corresponding prototype. Parent-level buttons
select a descendant for inspection, not a separate parent prototype. Scores are
softmax probabilities from the actual exported forward, not calibrated reliability.
Rotated EXIF images and transparent inputs are rejected rather than silently using
a different image interpretation. Folder and ZIP ingestion remain future work.

## Image placement

When the report contains angular t-SNE, the browser computes angular distances
from the image embedding to all prototype directions. It forms Gaussian affinities
at perplexity 30 over the nearest 90 prototypes, then minimizes conditional KL
against Student-t affinities in the fixed map. Four deterministic starting points
and a bounded, decreasing-loss line search reduce sensitivity to initialization.
Only the image moves. This follows the fixed-reference insertion approach used by
[openTSNE](https://opentsne.readthedocs.io/en/stable/api/index.html#opentsne.TSNEEmbedding.transform),
but is a separate implementation, not a port or an identical objective to a joint
refit. No t-SNE coordinates are assigned merely by selecting the predicted class.

The purple diamond shows the result. The status reports top-12 angular-neighbour
retention in the displayed neighbourhood and computation time. Low retention is a
limitation of the placement/map, not evidence that the model prediction is wrong.
A saved PCA mean and basis also support exact fixed-transform PCA placement of the
normalized embedding. Neither method changes the prototype coordinates.

## Authoring a distribution

Use the matching checkpoint and an ONNX export created with
`mt_export --include-embeddings`. This option preserves default export behaviour
when omitted; the backbone runs once and the small eval preclassification transform
is replayed to expose the embedding. Flat `Classifier` and `HierarchicalClassifier`
heads are supported. Other head families fail explicitly.

The browser currently supports one explicit preprocessing contract: repository
nearest-neighbour square uint8 resize, torchvision-style bilinear resize and centre
crop, then ImageNet RGB scaling and normalization. Supply this recipe to export
only after verifying it against the checkpoint's evaluation transform:

```json
{"contract":"nearest-square-uint8-bilinear-center-imagenet-v1","size":384,"resize":438}
```

The browser bilinear implementation can differ by one uint8 value from torchvision;
qualify end-to-end image results as well as identical-input runtime parity. This
contract does not cover arbitrary user transforms or dataset preprocessing overrides.

1. Export the checkpoint with the recipe, embeddings, and normal parity checks.
2. Generate the existing explorer from the same checkpoint (`mt_explore WEIGHTS
   --export --output REPORT`). Keep angular t-SNE enabled.
3. Unpack `onnxruntime-web@1.24.3` from npm. Add `LICENSE.txt` from the matching
   [ONNX Runtime release](https://github.com/microsoft/onnxruntime/blob/v1.24.3/LICENSE)
   to the unpacked package root; npm does not include this file.
4. Package the browser model beside the report:

```bash
uv run --no-sync python -m mini_trainer.visualization.prototype_space.browser \
  --export ONNX_BUNDLE --weights BEST_PT \
  --runtime UNPACKED_RUNTIME_PACKAGE --output REPORT/browser-model
```

5. Serve `REPORT` as static HTTP(S), and open `explorer.html`. The default bundle
   URL is `browser-model/manifest.json`. The bundle carries its checkpoint identity,
   class order, graph, external tensors, prototype directions, runtime and hashes.

The package is created atomically and existing destinations are not overwritten.
The runtime package and supplied ONNX export are trusted authoring inputs. Use only
trusted browser-bundle URLs: bundles also contain executable runtime assets.

## Hosting and limitations

Serve the explorer and browser bundle from the same origin, with correct HTML,
JavaScript, MJS and WASM MIME types. The initial implementation uses a worker from
the model-bundle origin, so a cross-origin bundle requires more than permitting
model-file CORS alone. No special COOP/COEP headers are needed for its single-thread
WASM configuration. A downloaded HTML retains numerical exploration, but browser
inference requires HTTP(S) serving and the model/runtime files; double-clicking the
HTML is not the complete inference distribution.

GBIF names and reference images are fetched directly by the browser, controlled
by **Settings → Class IDs**. Generic IDs make no GBIF requests. Species prediction
rows share attributed reference examples with the map and inspector; those photos
are not the user's query image. Missing names/images leave IDs and scores usable.
Optional packaged names work offline; remote photo URLs still require networking.
The precomputed numerical views do not depend on GBIF. The broader standalone-analysis roadmap is still open: this version does
not recompute global Ward/PCA/t-SNE diagnostics in the browser.

## Browser smoke check

With a Chromium executable available, the repository check opens the real hosted
URL, loads its model, selects the supplied local image, and requires predictions,
an embedding, and no uncaught page errors:

```bash
CHROME_PATH=/path/to/chrome node dev/prototype_space/check_inference.mjs \
  https://host/path/explorer.html /path/to/image.jpg /tmp/browser-check.json
```

Identical-input ONNX parity and original-image preprocessing comparisons remain
separate numerical qualification checks. The smoke check alone does not establish
accuracy or cross-browser compatibility. The pinned runtime module is distributed
with a `.js` extension because some static servers do not assign `.mjs` a JavaScript
MIME type; its contents are unchanged.

## Optional packaged GBIF names

For a GBIF-labelled report, prepare a resumable metadata snapshot before rendering
the release candidate. No credentials are needed. Each entry retains the original
ID and retrieval date; name resolution never changes the model vocabulary.

```bash
.venv/bin/python dev/prototype_space/prepare_gbif.py \
  --report /path/to/report/report-data.json \
  --output /path/to/report/gbif-snapshot.json
```

Use `--limit 100` for a bounded first pass; rerunning fills missing entries. Failed
lookups remain missing and can be retried. `render_report` embeds a neighbouring
`gbif-snapshot.json` in the HTML when present, so it survives downloading the report.
Re-render existing numerical data without recomputing geometry:

```bash
.venv/bin/python - /path/to/report <<'PY'
import sys
from pathlib import Path
from mini_trainer.visualization.prototype_space.explore import render_report
root = Path(sys.argv[1])
render_report(root, (root / "report-data.json").read_text())
PY
```

A snapshot explicitly supplies a GBIF default; saved user preferences take precedence.
The current helper packages taxon names, not image bytes or occurrence albums.
Static publication should include the snapshot as well as the regenerated HTML,
and regenerate its asset hashes. Keep a new release candidate separate from an
already published production release.
