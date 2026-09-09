# Test suite map

Run from the repository root with `bash dev/check.sh test tests/GROUP`.
Use `bash dev/check.sh all` for the complete static and runtime suite.
The shared harness hides CUDA by default; see [GPU validation](../dev/README.md)
before enabling hardware-specific cases. Test placement does not change markers,
optional dependencies, expected failures or slow-backbone requirements.

| Folder | Contract |
| --- | --- |
| `core/` | Configuration and public defaults |
| `data/` | Loading, workers, IO, dataset formats and augmentation |
| `modeling/` | Architectures, initialization, classifier shapes and embedding context |
| `training/` | Optimizers, losses, update counts and checkpoint state |
| `quantization/` | Native integer kernels, preparation, training models, materialization and PTQ/QAT |
| `export/` | ONNX export and native-quantized export contracts |
| `integration/` | Full training, lazy data and distributed integration |
| `benchmarks/` | Dataset/evaluation orchestration, inference probes, provenance and report storage |
| `logging/` | Console, TensorBoard and W&B logging |
| `utils/` | General device/plot helpers and the opt-in compatibility utility |

Shared test builders and state assertions currently live in the integration and
checkpoint modules that define their behavior. Imports and serialized test model
identifiers use those modules' new paths. Keep those paths importable for spawned
processes; avoid changing fixture semantics as part of directory cleanup.

The move preserved all 737 collected cases, with only test-module paths changed.
No tests were dropped, weakened or newly marked skipped to complete the grouping.
