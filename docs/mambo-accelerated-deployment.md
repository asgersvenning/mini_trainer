# Historical MAMBO preparation and precision comparison

This study qualified faster preparation and mixed precision before the later
streaming and TTA work. It explains the precision choice; use the
[current deployment guide](../deployment/README.md) for integration defaults and
[current comparisons](../deployment/README.md#release-comparison) for adoption.
Weights, standard ONNX graphs, presets and prediction interfaces were unchanged.

| Execution | Automatic precision in this study | FP32 reference |
|---|---|---|
| PyTorch CUDA | FP16 backbone; FP32 classifier and embeddings | Autocast disabled; matmul/cuDNN TF32 disabled |
| ONNX CUDA | TF32 execution of the standard FP32 graph | TF32 disabled |
| CPU, either backend | FP32 | FP32 |

BF16 passed subset qualification but was not evaluated on the full dataset.
FP16 had broader hardware support and smaller sampled changes from FP32.
Preparation kept interpolation arithmetic while using contiguous arrays and
computing only the retained crop. These combined changes were measured together;
the study does not isolate their individual contributions.

## Qualification

Both native FP16/BF16 and ONNX TF32 were qualified on the same seeded 4,096-image
subset against FP32. All predictive scores use the pinned mini_metrics revision,
threshold zero and no optimization. Prepared pixels were byte-identical on 256
real images plus frozen synthetic edge cases. Outputs and normalized embeddings
remain float32. Prediction/embedding modes agreed on the checked species labels;
custom-list filtering and same-batch embedding equivalence passed.

## Full evaluation and timing

Both accelerated backends completed all **58,640 Flemming images**, using the
same five presets and sample order as the FP32 reference. Every predictive metric
comes from `mini_metrics` commit `70cc69adc05362863439277048e06386c1f885e1`:
`threshold=0`, `optimal=False`, `simple=True`, `hierarchical=False`.
The primary results include all truth, including species outside the selected list.
Known-truth results retain 50,598 images at species level; membership is checked
separately at each rank. The [metric definitions](mambo-release-comparison.md#prediction-quality)
explain the different macro denominators, including predicted-only classes in F1.

Across five presets, three ranks and both truth populations, the largest
macro-accuracy difference from FP32 was below **0.094 percentage points**.
Northern-Europe macro-F1 over all truth was 0.2575 for v2, 0.2545 for v3 FP32,
and 0.2543 for both accelerated paths. No precision variant uniformly improved
every score. The figures and complete CSV retain the geographic comparisons.

![Macro-led species quality over all truth](assets/mambo-accelerated-quality-all.svg)

![Macro-led species quality over known truth](assets/mambo-accelerated-quality-known.svg)

The [complete metric CSV](assets/mambo-accelerated-metrics.csv) retains 108 rows /
756 scores: macro accuracy, precision, recall and F1, micro accuracy, Theil U and
coverage, at species/genus/family level for both populations and all five v3 lists.
The [compact evidence](assets/mambo-accelerated-comparison.json) includes hashes
and can regenerate the figures without access to the image dataset.

### Speed and memory

![Complete-pipeline throughput by batch size](assets/mambo-accelerated-speed.svg)

Northern Europe, **images per second** (higher is better):

| Pipeline | CPU batch 1 | CPU batch 8 | GPU batch 1 | GPU batch 8 | GPU batch 32 |
|---|---:|---:|---:|---:|---:|
| MAMBO v2 | 1.26 | 1.39 | 45.2 | 44.8 | 83.5 |
| Original v3 PyTorch FP32 | 6.74 | 7.39 | 27.1 | 46.6 | 44.5 |
| Updated v3 PyTorch auto | 5.70 | 7.84 | 29.8 | 126.7 | **136.2** |
| Original v3 ONNX FP32 | 9.64 | 9.71 | 30.4 | 42.8 | 42.4 |
| Updated v3 ONNX auto | 10.08 | 10.81 | 46.7 | 114.1 | **111.3** |

At GPU batch 32, PyTorch improved **3.06×** and ONNX **2.62×** over their original
pipelines. This was not a uniform CPU speedup: native batch-1 throughput fell
from 6.74 to 5.70 images/s. The early sequential pipeline plateaued around batch
8–32; that plateau is not a limit of the current streaming implementation.

![Peak host memory for the complete sweep](assets/mambo-accelerated-memory.svg)

| Updated runtime | CPU peak host RSS | GPU-run peak host RSS | CPU loading + first image | GPU loading + first image |
|---|---:|---:|---:|---:|
| PyTorch | 1,618 MiB | 2,512 MiB | 42.04 s | 42.96 s |
| ONNX | 689 MiB | 1,619 MiB | 0.37 s | 1.46 s |

Host memory increases versus the previous v3 sweep, while remaining below v2.
Native peak allocated GPU memory falls from 865 to **598 MiB** (v2: 1,936 MiB).
These are PyTorch allocator counters, not total device memory; an equivalent ONNX
allocator peak is unavailable. Host RSS includes loading and the entire batch
sweep. Startup excludes interpreter launch and explicit runtime setup. The native
startup cost included historical classifier initialization; see the
[FP32 report](mambo-v3-evaluation.md#startup-and-process-memory).

These compare the combined adapter changes against the recorded pre-change
pipeline on the same i7-12800H / RTX 3080 Ti Laptop GPU (16 GB), Linux/WSL2,
on AC power. They do not isolate AMP from preprocessing, and laptop conditions
vary between runs. Three fresh-process trials use the same seeded image bank,
four CPU threads, two warmups and seven observations per cell. Error bars show
the range of trial medians; reported throughput uses the median of all 21
observations. All speed figures include decoding, preparation, inference and
completed CPU results. Native CPU model threads were also explicitly set to four.
The v2 CPU result requires its documented caller-side float32 input cast.

## Evidence and replay

The [evidence JSON](assets/mambo-accelerated-comparison.json) contains the FP32
reference, automatic-precision results and source hashes. Regenerate its figures
and metric CSV without images or new inference:

```sh
.venv/bin/python -m dev.releases.mambo_v3.acceleration_report \
  --data docs/assets/mambo-accelerated-comparison.json \
  --output /tmp/mambo-acceleration-figures
```

The [historical report](https://github.com/asgersvenning/mini_trainer/blob/595582c212e7f248b400f7a782857bc5bdfe5111/docs/mambo-accelerated-deployment.md#reproduce)
retains original collection commands. Replaying them with today's adapter measures
a different pipeline. For new collection use the
[evaluation workflow](../dev/releases/mambo_v3/evaluation.md); for current platform
coverage use [final qualification](../dev/releases/mambo_v3/final-qualification.md).
