# Quantized training validation contract

This document describes the checks that must remain valid while quantization is
extended. Current results live in [benchmark findings](benchmarks.md); planned
work and target acceptance live in the [branch roadmap](quantization-roadmap.md).
Detailed historical runs are retained in the [experiment archive](https://github.com/asgersvenning/mini_trainer/blob/f5c69e7cab2bfde8a5467026b293858b93e628f9/docs/archive/benchmark-history.md).

## Model and dataset coverage

The primary model is EfficientNetV2-S with a symmetric hidden layer and normalized
`Classifier` or `HierarchicalClassifier`. Compare both heads on the same reviewed
Blair splits, with full and parameter-frozen training reported separately.
Synthetic oracle and MNIST cases verify fast integration. Synthetic 10k/100k-class
heads verify scaling; they do not replace real-data quality measurements.
Keep full million-class experiments off the laptop.

## Required invariants

- Quantization reports actual selected operators, physical weight/activation
  storage and floating operations. Current native training covers Linear weights
  and saved inputs; convolutions, gradients and optimizer state remain floating.
- Ordinary floating defaults, shape/dtype/device contracts, class ordering,
  normalized and hierarchical score semantics remain stable.
- SGD, AdamW and MuonAuxAdamW retain optimizer-step, scheduler, AMP-overflow,
  accumulation, compilation and checkpoint behavior. `_step_count` is part of
  this coordination; do not replace it without validating each optimizer's semantics.
- Controlled checkpoint continuation is tested with fixed ordering and disabled
  stochastic transforms. Arbitrary resume does not promise identical trajectories:
  general RNG/sampler state is not fully checkpointed.
- EMA remains disabled and unsupported; retain its explicit warning and strict
  expected-failure regression. DDP/FSDP support for native QT is not implied by
  floating CPU distributed tests.
- Native integer export and materialize-then-calibrate export are different
  numerical contracts. Verify output semantics, preprocessing, input identities,
  class mappings, operator placement and paired quality for the selected route.
- Worker selection respects available affinity/quota/allocation information;
  unrestricted shared-node contention still requires explicit conservative counts.

## Checks and interpretation

Use the [development checks](../dev/README.md) and grouped
[test suite](../tests/README.md). CPU checks cannot establish GPU correctness.
Intentional CUDA checks require explicit visibility and the documented flags;
optional runtimes/backbones may require a separately prepared environment.
Do not implicitly synchronize the working CUDA installation.

Benchmark quality uses `mini_metrics` Macro-F1, Macro-Recall, Macro-Precision,
Coverage and Theil's U at every reported hierarchy level. Keep paired seeds,
negative results, confidence/score semantics and undefined metrics visible.
Only compare resource measurements collected under suitable isolated conditions.
A few points of degradation may be acceptable when justified by a material
speed/cost or memory benefit. Record the per-profile acceptance limits before
qualification; integer execution alone is not acceptance.

The intended execution targets remain HPC GPU training on A40/A100/B300-class
systems with EPYC hosts, local training/fine-tuning and ONNX GPU inference on Spark
or the intended RTX desktop, and ONNX CPU inference on Raspberry Pi/ARM.
Support claims require evidence from the corresponding hardware and workload.
