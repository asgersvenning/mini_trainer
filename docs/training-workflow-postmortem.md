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

**Prepare the whole workflow before allocation.** The successful training started
about 97 minutes after allocation began. It finished around 12:35 the next day;
test inference completed at 16:11, metrics around 16:44 and archive verification
at 17:13, shortly before 17:24 expiry. These timestamps locate critical-path costs,
not a causal profile or proof that all qualification time was waste.

Commands were vulnerable to wrong working directories, missing YAML, torchrun
argument parsing, mixed checkout/package/PYTHONPATH identities and refused retries
into partial directories. One stage had weights/finite metrics but timed out during
remaining work. Record phase completion separately from process exit; a checkpoint
alone does not prove clean logger/export/teardown completion. Logs and atomic
completion records must survive terminal loss.

**Separate storage latency from compute.** Cold workers waited in `D` state /
`folio_wait_bit_common`; one 211-second window spent 146–150 seconds waiting for
input, and initial test inference reached only 69/2,473 batches after 43 minutes.
Repeated reads of an expert subset fell from hundreds of seconds to below one
second on WEKA. A disjoint-sample sweep favored 512 readers (~117 images/s median
confirmation); a larger-file workload provisionally favored 64. These are evidence
for latency hiding and cache sensitivity, not universal reader defaults or a
filesystem-internal diagnosis.

Production eventually stabilized near 650 training and 2,000–2,500 evaluation
images/s/GPU with 32 loader workers/rank. After staging, expert inference finished
in 2:49 and full test inference in 30:02. Separate encoded-byte read concurrency
from decode processes and CUDA transfer. Try meaningfully high bounded concurrency
when latency is evident, retaining errors, bytes, file sizes, sample coverage and
startup/drain costs. Disjoint paths are not guaranteed cold shared-cache data;
never clear shared caches to manufacture a benchmark.

**Use the qualified floating compute baseline.** Four-GPU warmed qualification:

| Batch/rank | Aggregate images/s | Peak allocated bytes (reported maximum) |
| --- | ---: | ---: |
| 32 | 1,059.648 | 8,332,350,464 |
| 64 | 1,741.601 | 15,640,267,776 |
| 128 | 2,272.305 | 30,251,580,416 |
| 256 | 2,517.928 | 59,508,456,960 |

128 → 256 gained ~10.8% throughput for nearly twice the allocation. Spare memory
invites a bounded test, not an assumption that larger global batches preserve
updates/schedule or improve convergence. Earlier single-GPU model compilation
reduced later-epoch time/allocation by roughly a third; optimizer compilation added
startup without steady-state gain, explicit prefetch had no convincing gain, and
combined INT8 runs timed out. Preserve these negative results; repeat only for a
changed mechanism/target. FP16 was tested, not proven superior to all BF16 recipes.

**Qualify required diagnostics and lifecycle together.** At 12,632 species,
confusion generation took ~8–13 s and warmed dendrogram rendering ~12–13 s; first
label resolution took roughly a minute in one run. Keep full-head figures and W&B
in the topology smoke, along with validation and save/reload. Preserve whole-matrix
inspection and explicit hierarchy-level selection instead of disabling useful
figures. A separate storage probe answers cold-IO questions.

**Triage warnings without expanding the campaign.** Early loss NaNs occurred in
both baselines and optional-feature trials; the final production audit was finite.
That weakens feature-specific attribution without proving harmlessness. Capture
first occurrence with sample/phase, dtype, features and finiteness; trigger bounded
replay only when needed. Compiler specialization, hierarchy scalar extraction and
DDP stride warnings merit changes when traces show meaningful repeated cost, not
simply to clean logs. EMA repair is separate work.

**Keep inference and packaging responsibilities clear.** Discovery must not filter
truth by model vocabulary or depend on training partitions. Distinguish taxonomy
transport failures from rank/index mapping errors. Derive inference configuration
from weights where reliable, with explicit legacy overrides. Preflight the metrics
environment and export tools before training; overlap independent preparation within
resource budgets. Package a small deployment subset separately from evidence and
optional resume history, retaining ONNX external tensors and explicit omissions.
Checksums establish integrity, not quality or complete provenance.

## Next-run minimum (planned)

Build on existing `dev/ucloud` setup/comparison/scaling/production helpers and normal
CLIs. Use a small manifest referencing current outputs, not a new workflow engine.

1. **Durable state/recovery:** resolved absolute commands, interpreters, package and
   harness identities, input hashes, stage times/logs, exit cause and completion
   checks. Reuse finished stages only when identities match; retain partial evidence.
2. **Preallocated preparation:** a tiny installed train → predict → mini_metrics →
   export → package fixture verifies dependencies and paths before renting GPUs.
   Keep W&B authentication interactive and secrets out of manifests.
3. **Separate qualification:** report startup, warm training, first-pass IO,
   validation/figures and teardown. Use one relevant topology smoke plus bounded
   storage calibration. Distinguish allocation deadline, stage limits and cleanup
   reserve; preserve the configured learning-rate schedule.

Acceptance includes interruption after preparation, failed child exit, timeout
after saving, logger teardown failure and completed-stage reuse. Keep normal CLI
execution, inspectable plans/status, explicit retry/resume, operator batch/worker
choices and a supported-checkpoint stop/continue decision. Do not promise
arbitrary-batch exact resume without RNG/sampler state.

Next, improve bounded read/staging calibration. Only build the
[optional prepared dataset](roadmap.md#optional-dataset-preparation-for-scalable-loading)
when total preparation plus expected reuse pays off. Preserve bytes, labels,
splits/order and failures; do not silently turn a partial prototype into the default.

Measure operator interventions, time to productive training, cold/warm throughput,
evaluation turnaround, repeated/failed stages and artifact size. Use tiny fixtures
for workflow contracts and target runs for hardware claims. Expand experiments only
when they change a decision; a production matrix is not a default test suite.
Unresolved PTQ/native integer and target-performance work belongs in the
[quantization roadmap](quantization-roadmap.md), not this operational plan.
