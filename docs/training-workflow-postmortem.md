# Production training: evidence and next-run priorities

Campaign: 10–11 September 2026. This is a historical assessment and proposed
workflow work, not implementation authorization. The quantization code is now
merged. Later MAMBO deployment qualification is recorded in the
[release handoff](../dev/releases/mambo_v3/final-qualification.md); its quality,
regional-list and threshold evidence supersedes the preliminary deployment
questions raised during training.

## What completed

| Evidence | Result and limit |
| --- | --- |
| Production recipe | EfficientNetV2-S, normalized hierarchical head, input 384; four full B200s, batch 256/rank, FP16 AMP, model compilation and 32 loader workers/rank. Optimizer compilation, explicit CUDA prefetch, INT8 training and EMA were off. |
| Training | 30 epochs in 17:34:23; training timer 16:27:22, evaluation timer 43:33; best epoch reported as 30. Timers omit some logging/teardown. |
| Quality | Validation species/genus/family micro accuracy 94.2353% / 97.6693% / 99.5047%, finite final losses. Preliminary test results agreed closely; expert data showed substantial domain/vocabulary shift. Use the [current deployment evidence](mambo-deployment-evidence.md) for comparisons. |
| Resume | Four ranks agreed on checkpoint hash, start epoch 3 and restoration of model/optimizer/scheduler/scaler. This is controlled restoration, not bit-exact arbitrary stochastic continuation. |
| Export | FP32 ONNX synthetic batches 1/2/4 passed at rtol/atol 1e-4, maximum absolute error 3.84e-5. ONNX PTQ used 128 training images and Percentile 99.9; graph/finite-output smoke passed, without establishing integer placement or quality. No quantized `.pt` was created. |
| Diagnostics/retention | Figures and W&B ran throughout production; 23.71 GiB before ZIP, reported ZIP integrity and archive checksum passed. |

Source: operator-provided campaign logs, tables and command output, not an
independent archive replay. Do not infer exact training source or model identity
from branch/version names; the later model card records remaining provenance gaps.
Successful floating training did **not** establish the original fully quantized
training objective or a controlled regression comparison against master.

The source splits contained 5,063,857 training, 633,224 validation and 632,913 test
images. Training count × 30 / training timer implies roughly 2,564 images/s
aggregate (641/rank), consistent with observed ~650/rank. Budget validation and
diagnostics separately rather than using the total source-row count as epoch size.

## Lessons that change the next run

**Prepare evaluation and recovery before allocation.** Training started about
97 minutes after allocation and finished at 12:35 the next day. Test inference,
metrics and archive verification finished at 16:11, 16:44 and 17:13 respectively,
against a 17:24 expiry. These timestamps show critical-path costs, not their causes.

Wrong working directories, missing YAML, torchrun parsing and mixed checkout/
package/PYTHONPATH identities caused interventions. Partial directories complicated
retries; one stage saved weights and finite metrics but timed out afterward.
A checkpoint therefore cannot stand in for completion of logging, export or teardown.

**Hide storage latency with bounded concurrency.** Cold workers waited in `D`
state / `folio_wait_bit_common`; a 211-second window spent 146–150 seconds waiting
for input. Initial test inference reached only 69/2,473 batches after 43 minutes.
Repeated expert-subset reads fell from hundreds of seconds to below one second on
WEKA. A disjoint-sample sweep favored 512 readers (~117 images/s median confirmation);
a larger-file workload provisionally favored 64. These support aggressive latency
hiding, not universal defaults or a filesystem-internal diagnosis.

Production stabilized near 650 training and 2,000–2,500 evaluation images/s/GPU
with 32 loader workers/rank. After staging, expert inference finished in 2:49 and
full test inference in 30:02. Allocate encoded-byte readers, decode processes and
CUDA transfer separately. Report errors, bytes/file sizes, sample coverage and
startup/drain costs. Disjoint paths do not guarantee cold shared-cache data; never
clear shared caches to manufacture a benchmark.

**Start from the qualified floating baseline.** Four-GPU warmed qualification:

| Batch/rank | Aggregate images/s | Peak allocated bytes (reported maximum) |
| --- | ---: | ---: |
| 32 | 1,059.648 | 8,332,350,464 |
| 64 | 1,741.601 | 15,640,267,776 |
| 128 | 2,272.305 | 30,251,580,416 |
| 256 | 2,517.928 | 59,508,456,960 |

128 → 256 gained ~10.8% throughput for nearly twice the allocation. Larger batches
also change update/schedule behavior; spare memory alone is not a convergence case.
Earlier single-GPU model compilation reduced later-epoch time/allocation by roughly
a third. Optimizer compilation added startup without steady-state gain, explicit
prefetch had no convincing gain, and combined INT8 runs timed out. Revisit these
negative results only for a changed mechanism or target. FP16 was tested, not
established as superior to all BF16 recipes.

**Include diagnostics and failure handling in qualification.** At 12,632 species,
confusion generation took ~8–13 s, warmed dendrogram rendering ~12–13 s, and first
label resolution roughly a minute in one run. Include full-head figures, W&B,
validation and save/reload in the topology smoke; storage calibration is separate.
Keep whole-matrix inspection and hierarchy-level selection.

Early loss NaNs occurred in both controls and optional-feature trials; the final
production audit was finite. This weakens feature-specific attribution without
proving harmlessness. Capture the first occurrence's sample, phase, dtype and
feature/finiteness state for bounded replay. Compiler specialization, hierarchy
scalar extraction and DDP stride warnings justify changes when traces establish
repeated cost. EMA repair remains separate.

## Next-run minimum (planned)

Extend the existing `dev/ucloud` helpers and normal CLIs; use a small stage manifest,
not another workflow engine. This work is still planned:

1. **Durable recovery:** record absolute commands, interpreter/package/harness
   identities, input hashes, phase times/logs, exit cause and atomic completion
   records. Keep plans/status inspectable and logs durable across terminal loss.
   Reuse completed stages only when identities match. Preserve partial evidence,
   explicit retry/resume, operator batch/worker overrides and checkpoint stop/
   continue controls. Do not promise arbitrary-batch exact resume without RNG and
   sampler state.
2. **Preallocated preparation:** run a tiny installed train → predict → mini_metrics
   → export → package fixture before renting GPUs. Keep authentication interactive.
   Inference discovery must preserve all truth regardless of vocabulary or training
   partition; distinguish taxonomy transport failures from mapping errors. Derive
   configuration from weights with explicit legacy overrides. Separate deployment
   assets, including ONNX external tensors, from evidence and optional resume history.
3. **Bounded qualification:** separate startup, warm training, first-pass IO,
   validation/figures and teardown. Use one relevant topology smoke and storage
   calibration; preserve the learning-rate schedule and distinguish allocation
   deadline, stage limits and cleanup reserve. Overlap independent work within
   resource budgets.

Acceptance must exercise interruption after preparation, failed child exit, timeout
after saving, logger teardown failure and reuse of completed stages. Track operator
interventions, time to useful work, evaluation turnaround and repeated/failed work.
Use tiny fixtures for workflow contracts and target runs for hardware claims.

Improve read/staging calibration before building the
[optional prepared dataset](roadmap.md#optional-dataset-preparation-for-scalable-loading).
Include preparation cost and expected reuse; preserve encoded bytes, labels,
splits/order and failures. PTQ/native integer and target-performance work belongs
in the [quantization roadmap](quantization-roadmap.md).

The campaign-specific staging/evaluation launchers remain in
[Git at 852bf71](https://github.com/asgersvenning/mini_trainer/tree/852bf712e85b8d1a6b9c9c6d31b3b5d807904303/dev/ucloud).
They used a source overlay pinned to `0c572ca` and job-specific `/work` paths.
Use maintained CLIs for new work; retiring those launchers does not invalidate the
512-reader and RAM-staging evidence above.
