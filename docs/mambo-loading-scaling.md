# Historical loading and scheduling diagnosis

After initial GPU acceleration, the synchronous adapter still prepared each
batch before running inference. Increasing batch size could not hide that
per-image cost. This led to the bounded streaming implementation described in
the [current pipeline review](mambo-inference-pipeline-review.md).

## Findings worth retaining

At four preparation workers, native batch-32 prepared-input throughput was
343.7 images/s versus 129.1 for the complete synchronous request. ONNX reached
203.3 versus 100.5. Prepared-input timing included transfers and CPU species
scores, but excluded loading and hierarchy; separately measured medians are
not additive.

One-batch lookahead improved native throughput from 128.2 to 154.2 images/s
and ONNX from 99.6 to 155.2. Eight workers helped native slightly but reduced
ONNX lookahead throughput to 132.7. The lesson is to budget preparation and
runtime threads separately and overlap stages, rather than assume more workers
or larger batches always help.

This was a warm-cache laptop experiment on 128 images, not an HPC worker-count
recommendation or a replacement for fresh-process release benchmarks.

## Evidence and replay

[Recorded measurements](assets/mambo-loading-scaling.json) retain all sweeps;
the [historical report](https://github.com/asgersvenning/mini_trainer/blob/852bf712e85b8d1a6b9c9c6d31b3b5d807904303/docs/mambo-loading-scaling.md)
records the method and commands for its retired diagnostic tools.
Use the [pipeline probe](../dev/releases/mambo_v3/pipeline-probe.md) and
[speed smoke](../dev/releases/mambo_v3/speed-smoke.md) for current behavior.
