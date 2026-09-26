# In-domain deployment evidence

Global vocabulary; 568,939 reporting images and 63,974 separate calibration images from the original 632,913-image test split. All truth is in vocabulary. Shared label-stratified 90/10 split, seed 42, grouped by image ID; per-rank mini_metrics Macro-F1 calibration, eps=0.01, quantiles, no bootstraps.

The recipe was selected on Flemming. These general photographs complement the more deployment-relevant Flemming monitoring crops; their different TTA response does not invalidate that deployment evidence.

Full support / >5 values use the same reporting rows. >5 requires truth and accepted-prediction support in every pipeline, separately per confidence setting. No rows are removed; per-class FP/FN remain intact.

## Species

| Pipeline | Confidence | Threshold | Macro accuracy (full / >5) | Macro-F1 (full / >5) | Coverage |
|---|---|---:|---:|---:|---:|
| MAMBO v2 | None | 0.0000 | 87.75% / 89.20% | 0.8504 / 0.8701 | 100.00% |
| MAMBO v2 | Calibrated | 0.3833 | 90.37% / 91.77% | 0.8577 / 0.8795 | 96.00% |
| V3 PyTorch | None | 0.0000 | 92.96% / 93.72% | 0.9145 / 0.9258 | 100.00% |
| V3 PyTorch | Calibrated | 0.5272 | 94.59% / 95.33% | 0.9207 / 0.9325 | 97.46% |
| V3 ONNX | None | 0.0000 | 92.95% / 93.72% | 0.9145 / 0.9258 | 100.00% |
| V3 ONNX | Calibrated | 0.5275 | 94.60% / 95.33% | 0.9208 / 0.9326 | 97.45% |
| V3 PyTorch + TTA | None | 0.0000 | 90.32% / 91.61% | 0.8931 / 0.9073 | 100.00% |
| V3 PyTorch + TTA | Calibrated | 0.3543 | 92.43% / 93.63% | 0.8991 / 0.9146 | 96.99% |
| V3 ONNX + TTA | None | 0.0000 | 90.33% / 91.62% | 0.8932 / 0.9074 | 100.00% |
| V3 ONNX + TTA | Calibrated | 0.3529 | 92.44% / 93.64% | 0.8991 / 0.9147 | 97.00% |

## Genus

| Pipeline | Confidence | Threshold | Macro accuracy (full / >5) | Macro-F1 (full / >5) | Coverage |
|---|---|---:|---:|---:|---:|
| MAMBO v2 | None | 0.0000 | 94.49% / 94.80% | 0.9214 / 0.9273 | 100.00% |
| MAMBO v2 | Calibrated | 0.5076 | 97.10% / 97.34% | 0.9316 / 0.9383 | 96.27% |
| V3 PyTorch | None | 0.0000 | 97.00% / 97.14% | 0.9600 / 0.9637 | 100.00% |
| V3 PyTorch | Calibrated | 0.6511 | 98.44% / 98.52% | 0.9671 / 0.9705 | 98.04% |
| V3 ONNX | None | 0.0000 | 97.00% / 97.14% | 0.9600 / 0.9636 | 100.00% |
| V3 ONNX | Calibrated | 0.6541 | 98.44% / 98.53% | 0.9671 / 0.9704 | 98.02% |
| V3 PyTorch + TTA | None | 0.0000 | 95.25% / 95.55% | 0.9422 / 0.9468 | 100.00% |
| V3 PyTorch + TTA | Calibrated | 0.4268 | 97.15% / 97.41% | 0.9493 / 0.9541 | 97.27% |
| V3 ONNX + TTA | None | 0.0000 | 95.24% / 95.54% | 0.9422 / 0.9468 | 100.00% |
| V3 ONNX + TTA | Calibrated | 0.4292 | 97.17% / 97.43% | 0.9494 / 0.9542 | 97.25% |

## Family

| Pipeline | Confidence | Threshold | Macro accuracy (full / >5) | Macro-F1 (full / >5) | Coverage |
|---|---|---:|---:|---:|---:|
| MAMBO v2 | None | 0.0000 | 97.06% / 96.97% | 0.9598 / 0.9602 | 100.00% |
| MAMBO v2 | Calibrated | 0.6285 | 98.86% / 98.83% | 0.9637 / 0.9650 | 98.52% |
| V3 PyTorch | None | 0.0000 | 98.47% / 98.43% | 0.9804 / 0.9798 | 100.00% |
| V3 PyTorch | Calibrated | 0.8646 | 99.49% / 99.47% | 0.9828 / 0.9823 | 98.98% |
| V3 ONNX | None | 0.0000 | 98.46% / 98.41% | 0.9800 / 0.9794 | 100.00% |
| V3 ONNX | Calibrated | 0.8654 | 99.49% / 99.47% | 0.9828 / 0.9823 | 98.98% |
| V3 PyTorch + TTA | None | 0.0000 | 96.81% / 96.72% | 0.9645 / 0.9644 | 100.00% |
| V3 PyTorch + TTA | Calibrated | 0.6618 | 98.85% / 98.81% | 0.9666 / 0.9665 | 97.72% |
| V3 ONNX + TTA | None | 0.0000 | 96.81% / 96.72% | 0.9645 / 0.9644 | 100.00% |
| V3 ONNX + TTA | Calibrated | 0.6640 | 98.85% / 98.81% | 0.9665 / 0.9664 | 97.70% |

## Support outside the truncated average

| Confidence | Rank | Shared classes | Truth images outside | Proportion |
|---|---|---:|---:|---:|
| zero | species | 10732 | 9,681 | 1.70% |
| zero | genus | 4077 | 1,912 | 0.34% |
| zero | family | 101 | 13 | 0.00% |
| optimized | species | 10602 | 10,680 | 1.88% |
| optimized | genus | 4042 | 2,143 | 0.38% |
| optimized | family | 101 | 13 | 0.00% |

Machine-readable [metrics](assets/mambo-indomain-tail.csv), [thresholds and split identities](assets/mambo-indomain-thresholds.json), and [class domains](assets/mambo-indomain-support.json) retain provenance and supplementary metrics. Thresholds are dataset-specific evidence, not new deployment defaults.

## Historical HPC timing boundaries

The latest V3 B200 timings are in the [current HPC evidence](mambo-hpc-evidence.md). The observations below predate the pipeline improvements.

The EPYC 9655/B200 campaign retains 3 fresh-process trials per variant/device, 7 request observations per cell and 3 streaming observations per cell. Global and northern-Europe timing presets are available. CPU runtime threads: 4; streaming preparation workers: 48; readers: 256. Request, streaming and prepared-input diagnostics have different boundaries; do not pool them. The short streaming bank contains 1,024 images and includes pipeline startup. Prepared-input diagnostics exclude decoding/hierarchy reduction but include transfers, and remain supplementary.

[Request observations](assets/mambo-indomain-speed.csv) and [streaming observations](assets/mambo-indomain-streaming-speed.csv) retain all trials. [Campaign provenance](assets/mambo-indomain-campaign.json) identifies source hashes and runtime environments. Peak host memory spans each complete benchmark process and its tested batch sizes; it is not per-cell model memory. V2 was tested through batch 32 on GPU, V3 through batch 256.

## Reproduce

Use the pinned mini_metrics environment described in the [UCloud workflow](../dev/releases/mambo_v3/ucloud-release.md). From the repository root, with the extracted archive beneath `local-evidence/ucloud-2026-09-25/`:

```sh
python -m dev.releases.mambo_v3.indomain_report \
  --root local-evidence/ucloud-2026-09-25/mambo-results/runs-transfers \
  --output local-evidence/ucloud-2026-09-25/presentation
python -m dev.releases.mambo_v3.indomain_speed \
  --source local-evidence/ucloud-2026-09-25/mambo-results/summary-transfers \
  --output local-evidence/ucloud-2026-09-25/presentation
```
