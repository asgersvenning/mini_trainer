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

The notebooks are historical demonstrations with saved illustrations, not qualified
end-to-end workflows. They contain IPython shell cells, paths relative to
`examples/`, CUDA/Spark assumptions and manual inference code predating the current
API. Dataset construction above runs from the root; do not execute notebook cells
by concatenating them as plain Python. For maintained training/evaluation examples,
use the [benchmark workflow](../dev/benchmarks/training.md); for model integration,
use the [deployment guide](../deployment/README.md).
