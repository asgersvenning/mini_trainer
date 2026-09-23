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

The release-facing [preset catalogue](../../../docs/model-presets.md) defines every
preset, its geographic filter, species count and evidence threshold. It includes
Australia (including Tasmania), Tasmania-only, the requested overlapping American,
Asian, African, Mediterranean and Arctic regions, plus Oceania, Southeast Asia,
East Asia and the Middle East. The northern-European scope lists ambiguous
historical additions in parentheses.

Presets aim to avoid most geographically nonsensical predictions while allowing
species that **can be found** in a region. They do not describe natural/native
distributions or where a species should occur. Introduced species, migrants and
vagrants are eligible; no establishment-status filter is applied. Exclusion does
not prove absence. Lists inherit metadata coverage and errors, sampling and
taxonomy. Changing allowed classes changes score normalization; confidence is
conditional on the selected list. The reconstruction details below concern the
two unchanged legacy presets; new filters and thresholds are defined in
[preset-definitions.toml](preset-definitions.toml).

New presets provisionally require **at least 3 regional metadata rows and at least
25 global rows**. Both minima are inclusive. This replaces the initial one-row
draft and remains subject to a final qualification decision. The snapshot already
has at least 50 global rows for every model species, so the global gate currently
excludes nothing further. Australia changes from 1,907 to 1,874 species, Tasmania
from 401 to 274, and Japan from 974 to 697. Decide whether to count distinct GBIF
observations before finalizing: multiple image rows are not necessarily independent
occurrence evidence. A local Tasmania check gives 274 species with either three
rows or three distinct `gbifID` values. Both legacy lists remain unchanged.

| Preset | Species | Construction and evidence | Limits |
| --- | ---: | --- | --- |
| `full` | 12,632 | Every species in the pinned model mapping | Global training vocabulary, not every Lepidoptera species |
| `europe` | 3,014 | Filter the pinned metadata by `continent == "EUROPE"`, count rows per `speciesKey`, retain counts **> 25**. Exact membership and retained count-table match. | Uses the metadata's continent assignment, not a union of entire countries. The upstream method that assigned continents is not established here. |
| `north_europe` | 1,977 | Filter `countryCode` to `DE DK EE FI LT LV NL NO PL SE`, count rows per `speciesKey`, retain counts **> 25**. Exact membership match to the weights and tagged `data/reduced.txt`. | This reconstructs the list but does not uniquely establish the original country expression: several additional countries leave membership unchanged. |

The ordered files in `presets/` contain GBIF species IDs, one per line. Their
source-weight paths and hashes are in `inventory.toml`. Extraction uses each
weight's `cls2idx` and active indices, preserving model order rather than sorting
IDs or inferring a list from the new evaluation data. Keep these release-versioned
memberships fixed for backwards compatibility. Custom lists should resolve IDs
explicitly and report missing/duplicate IDs and excluded truth labels.

### Reproduce construction from metadata

The user-identified Parquet is available locally even though the full image dataset
is not. [construction.toml](construction.toml) pins its SHA-256, byte size, geography
filters, counting rule and expected totals. Using the existing environment with
PyArrow available, run from the repository root:

```sh
.venv/bin/python -m dev.releases.mambo_v3.reconstruct_presets \
  examples/global_lepi/0032836-250426092105405_processing_metadata_postprocessed_quality_filtered.parquet
```

This reads only the geography/species columns after verifying the source hash and
compares reconstructed membership to the frozen lists. It does not alter their
model ordering. Count **metadata rows**, with no additional occurrence/image
deduplication, across **all existing splits `0`–`9`**, including held-out records.
Thus the historical README's phrase "training data" means the overall metadata
corpus for this reconstruction, not just the training partition. Preserve that
disclosure when reporting held-out metrics; the historical vocabulary selection
used those rows too. The script neither changes nor regenerates splits.

Europe selects 2,079,617 rows covering 3,132 species before the strict threshold.
All 3,132 per-species counts exactly match `tmp/europe_training_data.csv`; the
minimum included count is 26. That retained table's SHA-256 is
`47e817d3e5009f929df4a8bc7404e4434c98232806596bc1585dd8c2b88e37d5`.
The reconstruction no longer depends on those unversioned CSV/list files.
The world count table also matches all 12,632 species' metadata row counts.

The Europe filter includes records coded `TR` (743), `GE` (361), `AZ` (242),
`RU` (110,277) and `KZ` (13) **only when their continent field is `EUROPE`**;
it does not include all records from Turkey or the Caucasus. No Armenian records
pass this filter. Country-only selection cannot reproduce the preset using the
same >25-row rule: species `11470119` has 595 rows, all `ES`, and is included;
species `5145842` has 194 `ES` rows and is excluded (192 are `AFRICA`, two have
blank continent). Including all Spain would therefore force an unwanted species.

The northern reconstruction uses **Germany, Denmark, Estonia, Finland, Lithuania,
Latvia, Netherlands, Norway, Poland and Sweden**. These select 768,497 rows and
2,291 species before thresholding. Removing any one of those ten countries
changes membership. Adding the **UK (`GB`) adds 29 species** absent from the
frozen list. Adding Ireland (`IE`), Iceland (`IS`), Åland (`AX`), Faroe Islands
(`FO`), Guernsey (`GG`), Isle of Man (`IM`), Jersey (`JE`) and Svalbard/Jan Mayen
(`SJ`), individually or all together, changes no selected species. Consequently
the final list cannot tell us whether Ireland or Iceland was originally included.
These are tested equivalent additions, not an exhaustive enumeration of all
possible filters. No original generation script was recovered.

The source Parquet hash identifies the exact taxonomy/metadata snapshot used for
reproduction; it does not establish the original GBIF taxonomy retrieval date or
the upstream continent-assignment geometry. Keep those remaining provenance
limits explicit. Neither preset is a comprehensive regional checklist.
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
Wrapper input/error, CLI, masking and both new backend integration checks are now
implemented; see [deployment qualification](deployment-qualification.md). The
archived CSV schema alone is not proof of complete legacy CLI equivalence.
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

Portable assets and aligned PyTorch/ONNX adapters are implemented; see
[deployment qualification](deployment-qualification.md). Full Flemming metrics and
local CPU/GPU timings are documented in the
[measured release report](../../../docs/mambo-v3-evaluation.md), with reproduction
commands and the prepared UCloud handoff in [evaluation.md](evaluation.md).

The original in-domain split and taxonomy have been checked against all 632,913
archived test identities. Image verification and inference still require UCloud.
Archived selected predictions support historical context, not MAMBO_v2 model
quality or downstream embedding claims. Training-source revision and best-epoch
provenance remain unresolved; packaging checkout is not training provenance.

The [real-world v2/v3 comparison](../../../docs/mambo-release-comparison.md) adds
the published BioCLIP-2 model baseline across northern Europe, Europe and global,
with quality, speed and memory charts. Reproduction and the explicit ancillary
v2 CPU input adapter are documented in [release-comparison.md](release-comparison.md).
