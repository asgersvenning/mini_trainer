# Short UCloud speed check: full B200, then one MIG slice

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

On the second node, run the same command with `--output /work/mambo-speed/b200-mig`.
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

Expect minutes, with the small MIG slice potentially taking tens of minutes;
initial model downloads and cold storage add setup time. Each completed variant
prints a row and updates `summary.csv`. Runtime output and errors are in its `.log`.
Stop after these four variants on each node unless the results expose a specific
failure or unexplained regression.

**Return the two output folders**, or initially just both `summary.csv` and
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
