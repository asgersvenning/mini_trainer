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

## Focus a view

Use **Focus view** on a feature panel, or the global **View** selector, to open
spatial projection, dendrogram, distance matrices, class images, or local
neighbourhood inspection. Each focus keeps related panels together: for example,
projection includes the neighbourhood profile, local inspector, and class photos.
Choose **All views** to return to the overview.

Focused projection and dendrogram panels use the available viewport height as
well as width. Thumbnail settings collapse to leave more space for the map.
The spatial map adapts its aspect ratio while preserving equal geometric scale
on both axes; its raster follows display pixel density with readable screen-size labels.
Dendrograms use the wider layout with readable labels and internal scrolling on
narrow displays. Class selection, projection centre/plane, and the current tree
subtree survive view changes. Related panels wrap below the main feature.

## Class names and chance alignment

Open **Class names** to enable aliases in selection, neighbourhood profiles,
dendrogram leaves, local matrix axes, and neighbour tables. Explicitly enable
GBIF lookup when the class IDs are GBIF taxon keys. Names resolve asynchronously,
prioritizing visible labels and the selected neighbourhood; unresolved classes
retain their IDs. Online resolution requires the local server and GBIF access.
Search accepts IDs, already resolved names, or `row N` for a checkpoint row.

For offline aliases, import JSON such as `{"1837646": "Taxon name"}`, or
`{"by_id": {"1837646": "Taxon name"}, "by_index": {"0": "Another name"}}`.
Aliases affect display only and never change checkpoint class ordering or IDs.

The score selector switches between z-scores and **Chance-alignment p-value
(approx.)**, `q = Φ(−z)`. This is the approximate one-sided tail under independent
uniform directions on the unit hypersphere: small q means random directions
would rarely be this closely aligned. It is not a posterior probability of
similarity or a calibrated test of whether two learned classes differ.
The reference does not assume that the learned prototypes themselves are uniform.

Probability formatting uses a direct log-tail approximation; the local matrix
retains the repository's stored log-tail values. Logarithmic colour limits remain
configurable, and tooltips retain unclipped values. Histogram bins retain their
original z positions, while the neighbourhood probability profile uses a log
axis. Ward linkage heights keep their original units.

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
