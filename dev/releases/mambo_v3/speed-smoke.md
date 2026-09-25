# Short UCloud deployment speed check

Run from this checkout on each manually allocated node with `/work/datasets` mounted.
This measures **V3 PyTorch, ONNX, and each with default TTA**. No V2 run, quality
metrics, full-dataset preparation, campaign configuration or tuning sweep.

Use the existing working Torch and qualified ONNX environments. Pull the release
branch first; no reinstall is needed when those environments already exist.

```sh
export MAMBO_CACHE=/work/mambo-cache
.venv-mambo-runtime/bin/python -m dev.releases.mambo_v3.speed_smoke \
  --metadata /work/datasets/global_lepi/0032836-250426092105405_processing_metadata_postprocessed_quality_filtered.parquet \
  --onnx-python /tmp/mambo-ort-ptx/bin/python \
  --output /work/mambo-speed/b200-full
```

A MIG comparison is optional when it answers a specific deployment question.
For that comparison, use `--output /work/mambo-speed/b200-mig` on the second node.
Use a **single visible MIG slice**, not the entire parent GPU. `environment.json`
records `nvidia-smi -L`, visible-device settings and the CPU quota, so retain the
exact MIG profile when comparing results. The full node and slice may also differ
in their CPU allocation; this is a deployment comparison, not isolated GPU scaling.

The script selects the same 4,096 original test images, hashes/warms only those,
and downloads standard release weights if absent. Each variant uses batch 256,
the global list, auto precision, no embeddings, and three streaming passes.
Preparation workers follow the exposed CPU quota (maximum 48); override with
`--workers N` if the container does not expose the UCloud CPU allocation correctly.
Keep both runs on the same commit and runtime versions.

To compare the compact-preparation update with the completed baseline, rerun the
same command using a fresh output name such as `b200-full-compact`. No environment rebuild, new model download or campaign setup
is needed when the existing environments and model cache are available.

Expect minutes, with the small MIG slice potentially taking tens of minutes;
initial model downloads and cold storage add setup time. Each completed variant
prints a row and updates `summary.csv`. Runtime output and errors are in its `.log`.
Stop after these four variants on each node unless the results expose a specific
failure or unexplained regression.

**Return the output folder(s)**, or initially their `summary.csv` and
`environment.json` files. The JSON reports retain raw trial timings, runtime
versions, preparation counters, host memory, and Torch allocator peak GPU memory.
The primary comparison is streaming images/s. Request timing is a separate API
measurement; the prepared-input diagnostic is single-view even in TTA reports.
Memory figures are not all-backend GPU peak measurements. This is a warm-storage
speed check, not a quality evaluation or a measurement of cold WEKA throughput.

## Only if the new node needs environments

These commands reuse the previously successful split between Torch and ONNX;
they do not revisit runtime selection during this speed test.

```sh
uv venv --python 3.13 .venv-mambo-runtime
uv pip install --python .venv-mambo-runtime/bin/python --torch-backend=auto \
  -e '.[timm]' -e ./deployment pyarrow
uv venv --python 3.13 /tmp/mambo-ort-ptx
uv pip install --python /tmp/mambo-ort-ptx/bin/python \
  'onnxruntime-gpu[cuda,cudnn]==1.22.0' -e ./deployment
```

Use the ONNX version already qualified on B200 here. This is an environment-specific
test setup, not a new deployment-wide dependency pin. Existing environments need
none of these installation commands.

## Current preparation and host-overhead update

When the next full-B200 comparison is needed, reuse its working environments.
Native Torch decoding and compiled Pillow rotation are retained. Pixel selection
now gathers complete RGB pixels instead of using three-axis NumPy indexing. This
changes real preparation in both backends. The same stack removes Python-heavy
hierarchy cache keys, defers unused class-name dictionaries, reuses CPU interpolation
scratch, removes redundant score scans and fuses Torch normalization. Test these
together in a fresh directory:

```sh
export MAMBO_CACHE=/work/mambo-cache
.venv-mambo-runtime/bin/python -m dev.releases.mambo_v3.speed_smoke \
  --metadata /work/datasets/global_lepi/0032836-250426092105405_processing_metadata_postprocessed_quality_filtered.parquet \
  --onnx-python /tmp/mambo-ort-ptx/bin/python \
  --output /work/mambo-speed/b200-full-gather
```

No environment rebuild, model change or new setting is needed. Keep batch size
and worker settings unchanged. Run the four variants once; do not repeat the MIG,
quality or GPU-resident tests. The existing resident reference at batch 256 is
3,667 images/s; it excludes transfers and CPU result construction and remains a
reference, not an end-to-end promise. Keep prior output directories for comparison.

For a local check that removes model speed from the comparison, use the
[three-case pipeline probe](pipeline-probe.md). It does not require another model
evaluation or a new UCloud allocation.
