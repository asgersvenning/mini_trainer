# Continuous dataset benchmarks

[Browse benchmark runs and their summaries](https://github.com/asgersvenning/mini_trainer/actions/workflows/benchmarks.yml).
The workflow is defined in this repository; results start appearing there once the
change is merged and runs execute. Initial local evidence is recorded below.

The benchmark foundation exercises the actual training, checkpoint reload and
inference paths. CPU is a fast verification milestone, not the scope boundary.
The progression is synthetic oracle → MNIST → hierarchical Blair, with Birds and
iNaturalist 2021 reserved for larger, more expensive follow-ups.

Each run produces a visible Actions summary and a downloadable artifact containing
JSON reports, predictions, dataset/split manifests, configurations, checkpoints and
logs. Failures are retained alongside successful results. The artifact excludes
synthetic image files, which can be regenerated from the recorded seed; source
images for real datasets are never uploaded by the workflow.

Reports identify the source revision and source-content hash, lockfile hash,
installed versions, dataset/checkpoint hashes, seeds, class mappings, device,
precision, cache settings and explicitly exercised capabilities. Results marked
`completed` only establish that the real-data pipeline ran: they do not claim a
quality improvement. Synthetic profiles have an exact 100% oracle quality gate.

Artifacts are configured for 90-day retention. This is an initial reporting surface,
not a permanent model zoo. A durable results index/dashboard should archive these
versioned records before expiry, retain failures, and compare matching configurations
across revisions. It must separate quality from speed and avoid combining CPU/GPU,
precision, dataset, or dependency changes into a misleading trend.

## Initial local evidence

These are baseline observations, not proposed model-quality or speed improvements.
They use seed 42, one CPU thread, zero loader workers, no augmentation/EMA, and small
offline models. The synthetic runs use 12 epochs; real-data runs use 5 epochs.

| Profile | Held-out accuracy | Scope |
| --- | --- | --- |
| Synthetic CPU float32 | 100% | Exact repeatability verified within the local environment |
| Synthetic CUDA float32 | 100% | Strict deterministic algorithms; CUDA cache |
| Synthetic CUDA float16 | 100% | AMP and CUDA cache |
| Synthetic CUDA bfloat16 | 100% | AMP and CUDA cache |
| MNIST CUDA float16 | 97.12% | AMP, CUDA cache, explicit nondeterministic profile |
| MNIST CPU float32 | 97.3% | 4,000 train / 1,000 validation / 5,000 test |
| Blair CUDA float16 | 64.3% species / 80.4% parent | 3,704 train / 912 validation / 1,161 test |

GPU observations use an NVIDIA GeForce RTX 3080 Ti Laptop GPU. The environment
uses Python 3.13.7 and PyTorch 2.12.0. Single-run timing is diagnostic; it is not a
portable performance gate. CPU synthetic training-call wall time was about 2.2 s,
MNIST CPU about 11 s, and Blair GPU about 13.4 s. These scopes include setup,
training, validation, logging and checkpoints, but exclude dataset inventory and
final held-out inference. Runs were not a controlled throughput comparison.

Strict deterministic mode rejected CUDA adaptive-pooling backward in the real-data
backbone. The explicit `--allow-nondeterministic` profile permits those operations
and records that choice. Seeds and preserved inputs make the experiment repeatable;
they do not promise bitwise-identical GPU results. Neither AMP nor CUDA caching is
inference quantization or quantization-aware training.

## Coverage still needed

The reports currently mark EMA, distributed training, augmentation, quantization
and ONNX integration as unexercised by this benchmark suite. Existing focused tests
provide other evidence, but do not make those boxes true for these dataset runs.
Add explicit profiles and comparison criteria before claiming coverage or improvement.
The known EMA continuation failure remains recorded in the roadmap.

The plain AdamW/SGD step-tracking blocker discovered by this baseline is now fixed.
The trainer gates scheduler/EMA advancement on completed optimizer steps and the
native fused-optimizer AMP overflow signal. Real CPU and CUDA overflow regressions
cover MuonAuxAdamW, AdamW, SGD and fused AdamW/SGD. This establishes compatibility;
it does not yet measure their relative model quality. See the [step contract](../dev/README.md#optimizer-step-contract).

See the [benchmark guide](../dev/benchmarks/README.md) for local commands, GPU runner
configuration, real-data inputs and reproduction details.
