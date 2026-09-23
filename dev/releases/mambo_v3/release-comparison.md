# Compare MAMBO_v2 and MAMBO_v3

This comparison keeps northern Europe first, then Europe and global. It uses the
legacy geographic lists for the primary release comparison and separately shows
v3's updated Europe/northern-Europe lists. Full quality uses the existing Flemming
manifest, including out-of-vocabulary truth, and the same pinned metric policy.

## Historical model identity

The three MAMBO_v2 heads in `inventory.toml` have identical learned tensors and
metadata; only regional active indices differ. Their cached external BioCLIP-2
backbone is also required. `legacy_evaluation.py` verifies every historical Python
source file against commit `32b3cd661778356b2e8c4cff5b10fa9061aa6f5d`, validates head
hashes, checks shared parameters and pins both external backbone files by SHA-256.
The head hashes are observed retrieval hashes, not independent historical signatures.

Export the original source without switching this release branch:

```sh
mkdir -p /path/to/v2-source
git archive 32b3cd661778356b2e8c4cff5b10fa9061aa6f5d mini_trainer | tar -x -C /path/to/v2-source
```

Use a separate Python 3.13 environment. The measured environment reuses the
existing PyTorch 2.12.0+cu130 / torchvision 0.27.0 runtime and adds
`open_clip_torch==3.3.0`, `timm==1.0.25`, `huggingface_hub==0.36.2`,
`safetensors==0.6.2`, `ftfy==6.3.1`, `regex==2026.9.10`, and
`requests==2.34.2` with its dependencies. This is an isolated comparison environment,
not a recovered historical dependency lock or clean installation qualification.
Do not sync the repository environment or replace its CUDA wheels.

The local offline Hugging Face cache points `imageomics/bioclip-2` to snapshot
`2957b322090f9cb17ae72c71981c7218a28d81e0`. Required files:

| File | SHA-256 |
|---|---|
| `open_clip_config.json` | `1bf947e96e943fe50efd5c3e26c37f843a2fa3c358967719a68c8a6d17ce68c8` |
| `open_clip_model.safetensors` | `b7b2bf6fbc95799e42630e394cf95803892ab447c1a8ab629dbc82fbeaf7dfef` |

Keep `HF_HUB_OFFLINE=1` and select that cache using `HF_HUB_CACHE`. No downloads
belong in startup timing. Run the original source first on `PYTHONPATH` and use
`python -P`; otherwise the current checkout can silently shadow it.

## Qualification and full quality

```sh
HF_HUB_OFFLINE=1 HF_HUB_CACHE=/path/to/pinned-cache \
PYTHONPATH=/path/to/v2-source:/path/to/mini_trainer \
/path/to/v2-env/bin/python -P -m dev.releases.mambo_v3.legacy_evaluation qualification \
  --source /path/to/v2-source --weights /path/to/MAMBO \
  --manifest /path/to/flemming-manifest.json --root /path/to/flemming \
  --output /path/to/new-qualification
```

The 256-image qualification compares the shared-backbone collection path against
original API calls for every list and image. Full collection verifies the first
batch again. The original backbone and head computations remain unchanged;
features are reused across masks only after equality of the learned states is
established. Image bytes are checked against the manifest. Every original truth
label remains in the canonical CSVs.

Replace `qualification` with `full` for the full dataset, using
`--output /path/to/v2-full-phase/v2-full` for the chart aggregator's directory layout.
Alternatively, use `release_comparison full` with the shared arguments shown below.
For every list, run the
pinned metric environment using `dev.releases.mambo_v3.metrics --source ... --output ...`
as described in [evaluation.md](evaluation.md). `compare_quality.compare` with
`presets=["north_europe", "europe", "full"]` verifies paired v2/v3 identities,
truth and coverage and produces accuracy changes and label agreement.

## Speed and memory

`release_comparison.py benchmark` runs three fresh-process trials of v2 PyTorch
and v3 PyTorch/ONNX on CPU and CUDA, in alternating order. Arguments:

```sh
python -m dev.releases.mambo_v3.release_comparison benchmark \
  --v2-python /path/to/v2-env/bin/python --v3-python /path/to/v3-env/bin/python \
  --legacy-source /path/to/v2-source --legacy-weights /path/to/MAMBO \
  --hf-cache /path/to/pinned-cache --bundle /path/to/v3-bundle \
  --manifest /path/to/flemming-manifest.json --root /path/to/flemming \
  --output /path/to/new-comparison-timings
```

Do not overlap timing with other heavy work. Each configuration uses the same
seeded 32-image bank, four CPU threads, two warmups and seven observations per cell.
CPU batches are 1/8; GPU batches are 1/8/32. Calls include decoding, preprocessing,
hierarchy reduction and completed CPU results. Timings cover predictions only;
existing v3 embedding-mode evidence remains in the separate local report.

The first unadapted v2 CPU attempt is retained as failed evidence: its bfloat16
preprocessed input meets float32 convolution weights and raises
`RuntimeError: expected scalar type BFloat16 but found Float` in this environment.
The ancillary CPU benchmark uses `--cpu-float32`, a caller-side wrapper that casts
the original preprocessor's output to float32. It preserves its values and leaves
all historical source and learned weights unchanged. The orchestrator selects this
flag only for CPU, records it explicitly, and the CPU charts label the adapter.
It is not a shipped core fix. GPU and full quality use the original path.

V2 uses the original public API for GPU timing: an initial 512-pixel resize, BioCLIP
preprocessing to 224 pixels, and CUDA float16 autocast. V3 uses the qualified
384-pixel recipe and FP32. Both disable TF32. This is the intended real-world comparison of the models and pipelines shipped
in the two versions. Their resolution and precision choices explain the results. V2 has no qualified ONNX
variant in this comparison. The original source's loader warnings and expensive
classifier initialization are preserved; no core fix is applied here.

Reuse the unchanged earlier v3 global/updated-Europe results. The new v3 trials
add legacy Europe and both northern-Europe lists. Each plotted timing has three
trials; whiskers show trial-median range. RSS is the process high-water mark during
load and the batch sweep, using full-list-containing sweeps for all runtimes.
It is host memory, including initialization transients, not just weights or VRAM.
Native CUDA allocator peaks are separate; ONNX snapshots are not equivalent peaks.
Startup includes predictor construction and first completed call, with local cached
files. Process launch and explicit runtime setup are excluded; lazy imports during
model construction remain included. It is not a cold-boot measurement.

## Rebuild the charts

```sh
python -m dev.releases.mambo_v3.comparison_charts \
  --v2-quality /path/to/v2-full-phase --v3-quality /path/to/v3-full-phase \
  --v3-performance /path/to/original-v3-timings \
  --added-performance /path/to/new-comparison-timings --output /path/to/charts
```

Aggregation rejects unfinished runs, changed CSVs, different quality populations
or metric revisions, mismatched timing image banks and missing timing trials.
The resulting compact JSON excludes image identities and per-class predictions;
it records measurement summaries and source hashes. Regenerate shareable SVGs from
that JSON alone with `comparison_charts --data /path/to/mambo-release-comparison.json
--output /path/to/charts`. Keep raw evidence outside Git; charts and compact source
data are intentional release documentation assets.

In-domain comparison remains UCloud work using the original test split. This local
comparison neither changes thresholds/presets from test results nor publishes a
release. Core loading optimizations require a separate feature/fix branch.

The local run retains full v2 quality under
`local-evidence/mambo-release-comparison-quality/v2-full/` and successful timing
trials under `local-evidence/mambo-release-comparison-performance-cpu-adapter/`.
The original unadapted CPU failure remains in
`local-evidence/mambo-release-comparison-performance/trial-0-v2-cpu/report.json`.
Earlier v3 quality and timing inputs remain under `local-evidence/mambo-v3/`.

All 18 additional timing processes completed. An interrupted final v2 GPU trial
is retained separately and excluded; its successful replacement is
`trial-2-v2-cuda-0-retry1`. Every reported timing cell contains exactly three
successful trials. The original interrupted plan remains as `interrupted-plan.json`.

The revised charts use images/second throughout, with a shared GPU vertical scale.
Predictive scores come exclusively from mini_metrics (`micro_accuracy`, rather
than its macro `accuracy` field). All/known populations and macro-F1 class support
are defined in the report. Old metric JSON is retained beside its replacement as
`metrics-before-mini-metrics-only.json`; all 13 full-data extractions preserve the
previous accuracy and F1 values. Component timings are reaggregated from existing
benchmark evidence; no inference rerun is needed for these reporting corrections.

Run the metric integration regression with the pinned environment available:

```sh
MAMBO_METRICS_PYTHON=/path/to/metrics-env/bin/python \
  bash dev/check.sh all tests/releases
```

It exercises imbalanced classes and excluded truth to distinguish micro accuracy,
macro accuracy, all/known filtering and macro-F1 through the real mini_metrics API.
