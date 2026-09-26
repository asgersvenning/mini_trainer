# Historical V2/V3 comparison workflow

Reproduce the original Flemming comparison of V2's published pipeline and V3
FP32. It predates acceleration/TTA; use [current results](../../../deployment/README.md#release-comparison)
for adoption and the [UCloud runbook](ucloud-release.md) for a new campaign.
Run from the repository root. Replaying collection against today's adapter
measures a different pipeline; figures can be rebuilt directly from retained data.

## Historical model identity

The V2 runner verifies source commit `32b3cd661778356b2e8c4cff5b10fa9061aa6f5d`,
head hashes from [inventory.toml](inventory.toml), and external BioCLIP-2 hashes.
Heads share learned tensors and differ only by regional masks, allowing shared
features during collection. Head hashes record retrieval, not independent signatures.

Export the original source without switching this release branch:

```sh
mkdir -p /path/to/v2-source
git archive 32b3cd661778356b2e8c4cff5b10fa9061aa6f5d mini_trainer | tar -x -C /path/to/v2-source
```

Use a separate runtime and preserve the existing CUDA environment. The original
comparison used Python 3.13, PyTorch 2.12.0+cu130 / torchvision 0.27.0 and
open_clip_torch 3.3.0; the historical runbook linked below retains the full version
list. This was a contemporary comparison environment, not V2's original lock.

The offline Hugging Face cache must contain `imageomics/bioclip-2` snapshot
`2957b322090f9cb17ae72c71981c7218a28d81e0`. The runner's
[backbone verification](legacy_evaluation.py) pins both configuration and weights
by SHA-256. Download before timing, then set `HF_HUB_OFFLINE=1` and `HF_HUB_CACHE`.
Put the original source first on `PYTHONPATH` and use `python -P` so the current
checkout cannot shadow it. The [UCloud setup](ucloud-release.md#setup-with-uv)
automates source extraction and asset retrieval for that workflow.

## Qualification and full quality

```sh
HF_HUB_OFFLINE=1 HF_HUB_CACHE=/path/to/pinned-cache \
PYTHONPATH=/path/to/v2-source:/path/to/mini_trainer \
/path/to/v2-env/bin/python -P -m dev.releases.mambo_v3.legacy_evaluation qualification \
  --source /path/to/v2-source --weights /path/to/MAMBO \
  --manifest /path/to/flemming-manifest.json --root /path/to/flemming \
  --output /path/to/new-qualification
```

Qualification compares shared-backbone outputs with original API calls on 256
images for each list. Full collection rechecks its first batch and verifies image
hashes. Replace `qualification` with `full`, using output
`/path/to/v2-full-phase/v2-full` for the chart aggregator's layout.

Use the [pinned metric workflow](evaluation.md#metrics), retaining unknown truth.
[Metric definitions](../../../docs/mambo-release-comparison.md#prediction-quality)
cover macro/micro averages and all/known populations. Verify paired sample identity
and truth before interpreting changes.

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

The historical protocol uses the same seeded 32-image bank, four CPU threads,
two warmups and seven observations per cell: CPU batches 1/8, GPU 1/8/32,
predictions only. Do not overlap trials with other heavy work. The added V3 trials
cover legacy Europe and both northern lists, complementing the original
global/updated-Europe sweep. Keep adapter revisions consistent across those inputs.

Two distinctions are essential for interpreting the comparison:

- V2 CPU needed `--cpu-float32` because its bfloat16 inputs met float32 weights.
  The orchestrator selects this value-preserving input cast only for CPU;
  historical source/weights and the CUDA path remain unchanged.
- V2 uses an initial 512-pixel resize, BioCLIP's 224-pixel preprocessing and CUDA
  FP16 autocast; V3 uses 384 pixels and FP32. Both disable TF32. This compares
  release pipelines, not isolated backbones. V2 has no qualified ONNX variant.

[Shared timing boundaries](evaluation.md#timing-and-summary) define end-to-end,
startup and memory measurements. RSS uses full-list-containing sweeps; startup
includes construction and the first completed call with cached assets. Collection
wall time is not a backend speed comparison.

## Rebuild the charts

```sh
python -m dev.releases.mambo_v3.comparison_charts \
  --v2-quality /path/to/v2-full-phase --v3-quality /path/to/v3-full-phase \
  --v3-performance /path/to/original-v3-timings \
  --added-performance /path/to/new-comparison-timings --output /path/to/charts
```

Aggregation requires complete runs, unchanged metric inputs, matching populations,
metric revisions and image banks, and three timing trials per cell. It copies
`mini_metrics` scores at all ranks and both scopes; it computes no predictive
metrics. Keep raw predictions/observations outside Git and retain their hashes.

To regenerate retained figures and the complete metric CSV without private inputs:

```sh
.venv/bin/python -m dev.releases.mambo_v3.comparison_charts \
  --data docs/assets/mambo-release-comparison.json --output /tmp/mambo-fp32-figures
```

## Evidence and regression coverage

The original local evidence remains under:

| Evidence | Ignored location |
| --- | --- |
| Full V2 quality | `local-evidence/mambo-release-comparison-quality/v2-full/` |
| Successful additional timings | `local-evidence/mambo-release-comparison-performance-cpu-adapter/` |
| Unadapted V2 CPU failure | `local-evidence/mambo-release-comparison-performance/trial-0-v2-cpu/report.json` |
| Original V3 quality and timings | `local-evidence/mambo-v3/` |

These paths describe the retained local archive, not files shipped in the package.
Interrupted processes are excluded; each reported timing cell has three successful
trials. The [historical runbook](https://github.com/asgersvenning/mini_trainer/blob/b24a559b26849a57e83770335e6af00b08822807/dev/releases/mambo_v3/release-comparison.md)
retains individual retry and metric-migration details. Use the
[evidence policy](evidence-policy.md) when extending comparisons.

For changes to metric extraction, the focused integration regression checks
imbalanced classes, excluded truth, macro/micro averages and F1 through the real
pinned `mini_metrics` API:

```sh
MAMBO_METRICS_PYTHON=/path/to/metrics-env/bin/python \
  bash dev/check.sh test tests/releases/test_release_evaluation.py \
  -k pinned_metrics_distinguish_micro_macro_and_known_truth
```

For performance changes use the [pipeline review](../../../docs/mambo-inference-pipeline-review.md)
and [small speed check](speed-smoke.md), not a repeated full historical campaign.
