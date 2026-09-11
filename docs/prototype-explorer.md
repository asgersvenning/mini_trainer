# Prototype explorer

Explore a mini_trainer classifier's learned prototype geometry through linked
class-distance matrices, dendrograms, spatial projections, and optional GBIF
class photos. The viewer uses the repository's effective-weight handling and
class-distance functions. Projections and movable image labels do not change
prototype coordinates or neighbour rankings.

See the [compact development and research roadmap](prototype-explorer-roadmap.md)
for delivered capabilities and ordered next steps.

Install the optional dependencies in your intended environment:

```bash
uv pip install 'mini_trainer[explorer]' --torch-backend=auto
```

For an existing training environment, preserve its chosen PyTorch backend as
explained in the [installation guide](../README.md#installation).

## Open a model

```bash
mt_explore path/to/weights.pt
```

This opens a local browser page while the report is prepared. No output path,
class mapping, model architecture, or embedding dimension argument is required.
The information comes from the saved classifier metadata. The tool uses CPU
analysis and defaults to both angular t-SNE and PCA projections.

To select a file using the browser instead:

```bash
mt_explore
```

Choose **Model weights**, then **Open weights**. After preparation, click
**Open explorer**. The viewer's **Open another model** link returns to the picker.
The file is sent only to the local process, stored outside the served report
folder, and read with PyTorch's `weights_only=True` loader. The server binds to
loopback on an available port. Stop it with Ctrl+C when finished. Session files
are temporary; use **Save HTML report** to retain the portable numerical viewer.
Photos require the local server and network access to GBIF.

For SSH or an environment without a browser, use `--no-browser` and open the
printed URL through your usual local port forwarding. `--port` optionally fixes
the listening port. File selection reads from the browser's computer.

The equivalent module command works without an installed console entry point:

```bash
python -m mini_trainer.visualization.prototype_space path/to/weights.pt
```

## Export without serving

```bash
mt_explore path/to/weights.pt --export
```

The default output directory is `weights-explorer` in the current directory.
Use `--output /path/to/report` to choose another directory. Export writes
`explorer.html`, `report-data.json`, and `summary.json`; the HTML contains its
numerical data and scripts and can be opened offline.

For programmatic generation:

```python
from mini_trainer.visualization.prototype_space.launch import generate

generate('weights.pt', 'report')
```

Optional controls:

- `--pca-only`: skip angular t-SNE for a faster initial inspection.
- `--synthetic`: include independent vectors, planted groups, and algebraic
  edge cases as explicitly labelled comparison cases.
- `--threads 4`: bound CPU analysis threads (default: 4).

No dependencies are installed automatically and the source checkpoint is not
modified. Full pairwise analysis uses quadratic memory; a 12,632-class report
needs several GB of RAM and can take minutes. Synthetic cases probe the
calculations and rendering; they are not a prior for the learned geometry.

## Supported weights and interpretation

The loader currently supports floating-point **float32 linear heads** saved by
mini_trainer's `Classifier` and `HierarchicalClassifier`, including normalized
weight parametrizations and the checkpoint's class ordering. It reads the head
without constructing or downloading a backbone. Bias and norm ranges and the
checkpoint SHA-256 remain in report provenance.

Other head families, quantized encodings, and files without class metadata are
rejected explicitly. They require a defined prototype-extraction contract;
the explorer does not guess tensor keys or substitute raw parametrizations.
The supplied file should be the model weights file, not optimizer/training state.

See the [view guide](../dev/prototype_space/README.md#try-the-views),
[diagnostic API](prototype-diagnostics.md), and
[hyperspherical visualization review](hyperspherical-visualization.md) for
metric semantics, display clipping, projection distortion and thumbnail controls.
