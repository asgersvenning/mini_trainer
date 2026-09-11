# Quantization branch status

The goal is **not complete**: target-machine benefits have not been verified.
The maintained sources of truth are now:

- [Branch roadmap](quantization-roadmap.md): ordered work, bottlenecks, dependencies
  and completion gates.
- [Current benchmark findings](benchmarks.md): measured benefits, regressions and
  interpretation limits.
- [Command index](../dev/benchmarks/README.md): training, inference and reporting.
- [Historical evidence](https://github.com/asgersvenning/mini_trainer/blob/f5c69e7cab2bfde8a5467026b293858b93e628f9/docs/archive/benchmark-history.md): detailed experiments,
  unsuccessful approaches and numerical results retained for traceability.
- [Artifact retention](quantization-artifacts.md): what survived temporary cleanup
  and how to restore evidence.

Implemented locally: native INT8 Linear training with normalized symmetric heads;
optimizer/checkpoint safeguards; shared loading improvements; generic ONNX export;
explicit native-checkpoint materialization and calibration; CPU/TensorRT quality,
placement and resource commands; compact CPU/GPU records; opt-in release/Pages
publishing. The reporting service has not been qualified remotely.

Not established: worthwhile end-to-end training gains on HPC/desktop targets,
integer convolution training, distributed native QT, exact native-quantizer GPU
ONNX execution, or ARM production performance. EMA repair remains deferred.
Further notebook experiments, dashboard expansion, unrelated feature comparisons,
Birds/iNaturalist and dataset-format work are not prerequisites for this branch.
