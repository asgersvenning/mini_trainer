# MAMBO_v3 release inputs

This is the first release preparation increment, verified 2026-09-23. It freezes
inputs and compatibility expectations; it does not qualify a deployment adapter.
Model files, predictions and raw data stay outside Git in ignored storage.

## Reproduce the audit

`inventory.toml` pins 44 downloaded files by URL, relative path, size and SHA-256.
Download each URL to its corresponding path beneath an evidence root. Keep both
ONNX graphs with their adjacent `model.onnx.data`; a graph alone is incomplete.
Production files were checked against the published checksums; historical MAMBO
weights have newly observed hashes, not independent historical signatures.
`metadata_readback` identifies metadata hashed during this retrieval. These pins
establish reproducible inputs, not publisher authenticity.

From the repository root, using the existing environment without synchronization:

```sh
.venv/bin/python dev/releases/mambo_v3/audit.py local-evidence/mambo-v3 \
  --flemming /home/asger/data/flemming
.venv/bin/python dev/releases/mambo_v3/check_legacy_fixture.py
bash dev/check.sh test tests/releases/test_mambo_inventory.py
```

The audit checks every pinned file, safely reads checkpoint state on CPU, compares
all three ordered class mappings and both parent maps, recovers the region masks
from legacy weights and checks the committed lists. The standard ONNX manifest
must agree with the candidate mapping. It does not execute ONNX or build a model.
The legacy fixture extracts only the prediction container classes from the pinned
Git commit; it needs local Git history, but no old dependencies or backbone download.

Observed result: 44 files verified; all legacy/candidate class and parent mappings
identical; 12,632 species, 4,476 genera, 104 families. No additions, removals or
index remappings. This does not imply equal predictions: the backbone changes from
BioCLIP-2 to EfficientNetV2-S and preprocessing changes from 512 to 384 pixels.
The old files contain classifier weights and require a separately available
backbone; they are not standalone offline baseline bundles.

## Regional scope and construction

The presets restrict possible species predictions. They are not geographic
boundaries, exhaustive regional checklists, locality detection, or guarantees
that an excluded species cannot occur in the region. They inherit the coverage,
sampling and taxonomy of the training data. Changing the allowed classes also
changes score normalization; confidence is conditional on the selected list.

| Preset | Species | Construction and evidence | Limits |
| --- | ---: | --- | --- |
| `full` | 12,632 | Every species in the pinned model mapping | Global training vocabulary, not every Lepidoptera species |
| `europe` | 3,014 | MAMBO_v2 README: species with **more than 25** training records in Europe. Recovered from `classifier.active_indices` in the legacy Europe weights. | The precise geographic boundary/country set, occurrence query, deduplication and counting unit are not established by the tagged README. |
| `north_europe` | 1,977 | Recovered from the northern legacy weights; exact membership equals `data/reduced.txt` at the pinned tag. A subset of `europe`. | The original geographic definition, source checklist/query and inclusion threshold have not been recovered. Do not describe it as a comprehensive northern-European checklist. |

The ordered files in `presets/` contain GBIF species IDs, one per line. Their
source-weight paths and hashes are in `inventory.toml`. Extraction uses each
weight's `cls2idx` and active indices, preserving model order rather than sorting
IDs or inferring a list from the new evaluation data. Keep these release-versioned
memberships fixed for backwards compatibility. Custom lists should resolve IDs
explicitly and report missing/duplicate IDs and excluded truth labels.

Additional local corroboration: `tmp/europe_gbif_id_list.txt` matches the Europe
membership exactly. Filtering `tmp/europe_training_data.csv` by
`europeFrequency > 25` reproduces all 3,014 IDs from 3,132 table rows; the minimum
included count is 26. Its SHA-256 is
`47e817d3e5009f929df4a8bc7404e4434c98232806596bc1585dd8c2b88e37d5`.
These unversioned local files corroborate the threshold but do not establish the
missing original geographic query. They are not required by the release audit.

Before claiming independently reproducible geographic construction, recover and
publish the source dataset/checklist version, region geometry or country set,
query/filter code, counting unit, threshold, taxonomy version and retrieval date.
A future regenerated list should have its own revision and added/removed-ID report;
it must not silently replace these compatibility presets. Deployment documentation
and API preset metadata should expose count, membership, rule and provenance gaps.

## Compatibility boundary

Pinned baseline: MAMBO_v2, commit
`32b3cd661778356b2e8c4cff5b10fa9061aa6f5d`, plus the observed weight hashes.
The tagged `mini_trainer/deploy.py`, `mini_trainer/classifier.py` and
`mini_trainer/hierarchical/model.py` establish these expectations:

| Surface | Preserve / qualify |
| --- | --- |
| Python entry | `mini_trainer.deploy.Predictor(device="cuda", model=None, weights=None, class_mask=None, **kwargs)`; Europe default; `model` and `weights` mutually exclusive |
| Calls | `predict(x, **kwargs)` and `__call__`; path, NumPy, tensor or iterable; CHW/BCHW and grayscale handling; iterable stacked as one batch historically |
| Masks | Species ID lists or index masks; `-1` clears the mask; region selection is distinct from class ordering |
| Result | Iterable/indexable hierarchical prediction; top-1 item has native tuples `label`, `confidence`, `index`, ordered species/genus/family; `to_dict()` returns a list of dictionaries |
| Embeddings | `predict_with_embeddings` returns `(prediction, embeddings)`; new embedding dimension is model-specific, not BioCLIP-compatible |
| CLI | Restore `mambo_predict`, `--model`/`-M`, explicit weights and result-name convention; test parsing and exported rows before claiming compatibility |
| Top-k | Legacy ranks are independently ranked; they are not necessarily one ancestral path. `topk>1` warns as experimental; exceeding the smallest rank width raises. Legacy nested-result serialization is incomplete. Define supported behavior explicitly rather than perpetuating defects. |

`compatibility.toml` contains a small captured top-1 fixture and archived evaluation
CSV columns. The executable fixture checks the original container behavior only.
Wrapper input/error, CLI, masking and both new backend integration tests remain to
be implemented. The archived CSV schema alone is not proof of legacy CLI equivalence.
The legacy probability-detection heuristic uses a batch-wide sum; do not enshrine
that defect as a new probability contract. Any shared core correction belongs on a
feature/master branch before merge into the release branch.

## Evaluation handoff

Local Flemming has **58,640 images / 522 species**. Species-directory and filename
identities match every archived expert species-level prediction, with no missing
or extra JPEGs. This is membership evidence, not image-content checksum equality.
Both regional presets exclude 16 truth species / 8,042 images here. Preserve them
in evaluation and report all-image and in-vocabulary metrics separately; do not
silently drop unknown labels to improve accuracy.

The global in-domain dataset is intentionally absent locally. Run its comparison
on UCloud where `/work/global_lepi` is available, using the original supplied test
split and taxonomy. The pinned training config identifies the original Parquet;
`evaluation/in-domain/provenance/staging.json` maps staged filenames back to source
images. Keep that mapping when joining the archived predictions; numeric staged
names must not be treated as original image identities. Verify the supplied split,
source membership and expected 632,913 predictions before a full run, with a small
qualification first. Do not regenerate a random split from the training proportion.

Next implementation: self-contained portable assets and aligned PyTorch/ONNX
adapters, followed by a small Flemming qualification, then full local task metrics
and CPU/laptop-GPU timings. Use the same versioned runner/config on UCloud for
in-domain results. Archived selected predictions support a historical baseline,
not new-backend top-k or embedding quality claims. No fresh inference or speed
benchmark was performed in this increment. Training-source revision and best-epoch
provenance remain unresolved; packaging checkout is not training provenance.
