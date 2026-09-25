# Historical FP32 batch-scaling diagnosis

At adapter revision `a99b855`, V3's plateau had two causes: serial, allocation-heavy
CPU preprocessing and strict FP32 backbone execution. Batch sizes reached the
model correctly. This diagnosis motivated the implemented
[preparation and mixed-precision changes](mambo-accelerated-deployment.md);
it does not describe the current pipeline.

## Findings worth retaining

- Advanced indexing produced non-contiguous arrays, interpolation promoted
  intermediates to float64, and the adapter resized pixels later discarded by
  the crop. Contiguous intermediates plus computing only the retained crop
  reduced preparation of 32 images from 562 to 300 ms in the controlled probe.
- With GPU-resident input, native batch-32 throughput rose from 177.6 images/s
  in FP32/NCHW to 371.4 with FP16/NCHW and 418.4 with FP16/channels-last.
  Layout alone did not help. Convolutions, batch normalization and SiLU dominated;
  the classifier contributed about 0.15% of kernel time.
- V2 used a different backbone at 224 pixels with autocast; V3 used 384 pixels
  and FP32. Neither model size alone nor a larger batch predicts relative speed.
- ONNX convolution ran on CUDA. Small CPU graph nodes were not evidence of a
  silent CPU backbone fallback.

These are controlled laptop interventions, not current release benchmarks.
They establish preparation and precision effects, not a hardware-counter
distinction between arithmetic and memory-bandwidth limits.

## Evidence and replay

[Recorded measurements](assets/mambo-batch-diagnosis.json) retain the sweeps.
Replay the retired probes with the matching historical adapter; the
[historical report](https://github.com/asgersvenning/mini_trainer/blob/852bf712e85b8d1a6b9c9c6d31b3b5d807904303/docs/mambo-batch-scaling.md)
records commands, environments and methodology. Do not apply its monkey patches
to current code.

For current work use the [pipeline review](mambo-inference-pipeline-review.md),
[pipeline probe](../dev/releases/mambo_v3/pipeline-probe.md) and
[target speed check](../dev/releases/mambo_v3/speed-smoke.md).
