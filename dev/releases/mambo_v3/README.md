# MAMBO_v3 release maintenance

Start with the [deployment README](../../../deployment/README.md) for integration.
This directory maintains release inputs, qualification and reproducible evaluation.
Models, predictions and raw datasets stay outside Git.

| Task | Maintained source |
| --- | --- |
| Build and qualify release artifacts | [Deployment qualification](deployment-qualification.md); [final qualification](final-qualification.md) records the completed candidate |
| Prepare publication | [Publication checklist](publication.md) |
| Evaluate or regenerate reports | [Evaluation workflow](evaluation.md), [UCloud runbook](ucloud-release.md), [evidence policy](evidence-policy.md) |
| Inspect or revise geographic presets | [Catalogue](../../../docs/model-presets.md), [definitions](preset-definitions.toml), [generated hashes/counts](preset-manifest.toml) |
| Integrate or migrate from V2 | [Integration details](../../../docs/mambo-integration.md) |
| Verify model identity and training provenance | [Inventory](inventory.toml), [model provenance](model-provenance.toml), audit below |

## Reproduce the audit

`inventory.toml` pins 44 downloaded files by URL, relative path, size and SHA-256.
Download each to its corresponding path beneath an evidence root. Keep both ONNX
graphs beside their `model.onnx.data` files. Production hashes were checked against
published checksums; legacy weight hashes were observed during retrieval, not
independently authenticated. `metadata_readback` identifies metadata hashed then.

From the repository root, using the existing environment without synchronization:

```sh
.venv/bin/python dev/releases/mambo_v3/audit.py local-evidence/mambo-v3 \
  --flemming /home/asger/data/flemming
.venv/bin/python dev/releases/mambo_v3/check_legacy_fixture.py
bash dev/check.sh test tests/releases/test_mambo_inventory.py
```

The audit verifies pinned files, safely reads checkpoint state on CPU, compares
ordered class/parent mappings, recovers legacy masks and checks the ONNX manifest.
It does not execute models. The container fixture needs the pinned local Git
history, but neither old dependencies nor a backbone download.

The September 23 audit verified all 44 files and identical V2/V3 mappings:
12,632 species, 4,476 genera and 104 families, with no index remapping. Predictions
are not equivalent: BioCLIP-2 becomes EfficientNetV2-S and input size changes
from 512 to 384. Legacy classifier weights need a separately available backbone.
Best epoch 30 is verified; the exact training revision and original
`initial_seed42.pt` hash remain unavailable. [Final qualification](final-qualification.md)
distinguishes the recovered initialization recipe from verified starting bytes.

## Regional scope and construction

The [preset catalogue](../../../docs/model-presets.md) owns current geographic
scope, thresholds and membership rules. The details here explain how the unchanged
legacy lists were recovered; [construction.toml](construction.toml) pins the source
Parquet hash, filters and totals. Presets follow model order from each weight's
`cls2idx` and active indices, not sorted GBIF IDs.

### Reproduce construction from metadata

With PyArrow installed in the existing environment, run:

```sh
.venv/bin/python -m dev.releases.mambo_v3.reconstruct_presets \
  examples/global_lepi/0032836-250426092105405_processing_metadata_postprocessed_quality_filtered.parquet
```

The script verifies the Parquet hash and frozen memberships without changing
ordering or splits. Counts use metadata rows, without additional deduplication,
across **all existing splits 0–9, including held-out records**. Historical vocabulary
selection therefore used test rows too; retain this disclosure in evaluation.

| Legacy preset | Reconstruction | Evidence / limit |
| --- | --- | --- |
| `europe` | `continent == "EUROPE"`; species with >25 rows | 3,014 species; membership and all 3,132 pre-threshold species counts match the retained historical table. |
| `north_europe` | `countryCode` in `DE DK EE FI LT LV NL NO PL SE`; species with >25 rows | 1,977 species; exact membership match, but the original country expression is not uniquely recoverable. |

Europe cannot be reproduced as a union of entire countries with the same threshold.
For example, species `11470119` is included with 595 Spanish rows, while `5145842`
is excluded despite 194 Spanish rows (192 labelled AFRICA, two without continent).
Transcontinental countries contribute only their EUROPE-labelled records. The
original continent-assignment geometry and taxonomy retrieval date are unknown.

For northern Europe, removing any of the ten countries changes membership; adding
the UK adds 29 unwanted species. Adding any or all `IE IS AX FO GG IM JE SJ` changes
nothing. These are tested equivalent additions, not a recovered original script or
an exhaustive list of possible filters. The catalogue marks this ambiguity.

The historical Europe count-table SHA-256 is
`47e817d3e5009f929df4a8bc7404e4434c98232806596bc1585dd8c2b88e37d5`;
reconstruction no longer depends on that unversioned table. Published memberships
remain fixed; intentional changes require a new revision and added/removed-ID report.

## Compatibility boundary

The baseline is MAMBO_v2 commit `32b3cd661778356b2e8c4cff5b10fa9061aa6f5d` plus
observed weight hashes. [compatibility.toml](compatibility.toml) captures its top-1
container and archived CSV columns. The fixture qualifies that container only;
current input, masking, CLI and backend coverage is described in
[deployment qualification](deployment-qualification.md).

Keep these historical distinctions when interpreting compatibility:

- V2 defaulted to Europe; portable V3 defaults to global. Embeddings and predictions
  change with the model; [migration guidance](../../../docs/mambo-integration.md#moving-from-v2)
  owns current calling conventions and output differences.
- Legacy ranks were independently ranked, not necessarily one ancestral path.
  Top-k beyond one was experimental and nested-result serialization incomplete.
- The legacy probability heuristic used a batch-wide sum. This defect and an
  archived CSV schema are not sufficient grounds for promising score or CLI parity.

## Historical dataset quality filter

The [retired filtering script](https://github.com/asgersvenning/mini_trainer/blob/027e5b8a6e82b0356c28ea47672d1b69bdea0a7b/examples/apply_quality_filter.py)
records this policy: retain images predicted as `Valid` or `Dead` with confidence
≥0.5, then retain species with ≥50 surviving images. It selects Parquet records by
matching filename stems to the selected image IDs. This dataset filter is separate
from the regional preset occurrence thresholds above.

The script was labelled reference-only; its remote I/O is not a maintained rebuild
workflow, and it does not prove how the supplied Parquet bytes were produced.
Release construction uses the hashed metadata snapshot in `construction.toml`.

## Evaluation handoff

Flemming contains 58,640 images / 522 species. Its species-directory and filename
identities match the archived expert predictions; this is membership evidence,
not image-content checksum equality. Both legacy regional lists exclude 16 truth
species / 8,042 images. Keep those rows visible in evaluation.

The 632,913-image global-lepi test split was evaluated on UCloud without resplitting.
Preserve original sample identities when joining numeric staging filenames.
Retained prediction/confidence archives support metric recomputation without images.

Current results: [Flemming](../../../docs/mambo-deployment-evidence.md),
[in-domain](../../../docs/mambo-indomain-evidence.md), [HPC timings](../../../docs/mambo-hpc-evidence.md).
Use the workflows linked above; historical first-pass reports do not describe the
current default-TTA comparison.
