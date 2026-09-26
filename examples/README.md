# Examples

Use the [root installation guide](../README.md) first. Run dataset constructors
from the repository root in the existing environment:

```sh
.venv/bin/python -m examples.mnist.construct
.venv/bin/python -m examples.blair.construct
```

| Example | Purpose |
| --- | --- |
| [MNIST notebook](mnist.ipynb) | Flat classification, a constructed long tail and prediction plots; constructor keeps at most 500 images per digit in each original split. |
| [Blair notebook](blair.ipynb) | Hierarchical specimen classification and visualization; constructor preserves training/testing folders. |
| [Bird dataset constructor](birds/construct.py) | Downloads the published train/valid/test archives through Hugging Face. |
| [iNaturalist constructor](inat2021/construct.py) | Downloads mini/full training and validation data, resolves GBIF taxonomy and writes a data index. |

Constructors accept `--output_dir` and use a `.complete` marker; without it they
remove partial split directories before rebuilding. Use a dedicated destination.
The iNaturalist destination additionally appends `mini` or `full`.

Open notebooks with the existing environment's Python kernel, starting in the
repository root or `examples/`. The first cell constructs data from the root, then
sets `examples/` as the working directory. Choose either single-node or Spark
training; those cells retain their CUDA assumptions and are not a fresh training
qualification. Batched Python inference uses checkpoint mappings and actual image
paths, with CPU fallback. Saved plots are historical illustrations, not new results.

For maintained training/evaluation benchmarks use the
[benchmark workflow](../dev/benchmarks/training.md); for MAMBO model integration,
use the [deployment guide](../deployment/README.md).
