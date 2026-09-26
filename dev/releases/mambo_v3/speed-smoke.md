# Short UCloud deployment speed check

Use a full B200 node with `/work/datasets` mounted. The fresh-node workflow is:

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/):

   ```sh
   curl -LsSf https://astral.sh/uv/install.sh | sh
   source "$HOME/.local/bin/env"
   ```

2. Clone the release branch:

   ```sh
   cd /work
   git clone --branch release/mambo-v3 https://github.com/asgersvenning/mini_trainer.git
   cd mini_trainer
   ```

3. Create and activate the environment, then install its dependencies:

   ```sh
   uv venv --python 3.13 .venv-mambo-runtime
   source .venv-mambo-runtime/bin/activate
   uv pip install --torch-backend=auto -e '.[timm]' -e ./deployment pyarrow
   ```

4. Run the experiment:

   ```sh
   export MAMBO_CACHE=/work/mambo-cache
   python -m dev.releases.mambo_v3.speed_smoke \
     --metadata /work/datasets/global_lepi/0032836-250426092105405_processing_metadata_postprocessed_quality_filtered.parquet \
     --output /work/mambo-speed/b200-full-gather
   ```

The command uses uv to prepare/reuse the B200-qualified ONNX Runtime 1.22.0 in
`$MAMBO_CACHE/speed-onnx-1.22`, separately from the active Torch environment.
This is specific to the B200 experiment, not a deployment-wide version pin.
To reuse an existing ONNX interpreter instead, supply `--onnx-python /path/to/python`;
that bypasses automatic environment setup. Model assets download automatically.

The test selects and warms the same 4,096 original test images, using batch 256,
the global list, auto precision and no embeddings. It runs **PyTorch, ONNX, and
both with default TTA**, with three streaming passes per variant. Preparation
workers follow the CPU quota, capped at 48; `--workers N` overrides this if needed.
Keep sample, batch and worker settings unchanged for the pipeline comparison.

Initial dependency/model downloads and cold storage add setup time. Each completed
variant prints a row and updates `summary.csv`; runtime output/errors are in its
`.log`. Outputs remain under `/work`. Choose a fresh output directory each run.
No campaign configuration, full-data hashing, quality evaluation or MIG rerun is
needed. A MIG comparison is optional when it answers a specific deployment question.

Return the complete reports after completion:

```sh
cat /work/mambo-speed/b200-full-gather/summary.csv
tar -czf /work/mambo-speed/b200-full-gather.tar.gz \
  -C /work/mambo-speed b200-full-gather
```

The archive retains raw timings, preparation counters, runtime versions, CPU/GPU
identity, host memory and Torch allocator peak GPU memory. Streaming images/s is
the primary comparison. Request timing is a separate API measurement; the
prepared-input diagnostic is single-view even in TTA reports. Memory figures are
not all-backend GPU peaks. This is a warm-storage speed check, not cold WEKA throughput.
The existing resident reference at batch 256 is 3,667 images/s; it excludes transfers
and CPU result construction and remains a reference, not an end-to-end promise.

For targeted local diagnosis, the [pipeline probe and stage traces](pipeline-probe.md)
separate preparation, transfers and results from model execution. They are not
additional required steps for this experiment.
