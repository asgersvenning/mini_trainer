# Local release evaluation

This runbook collects same-model PyTorch/ONNX evidence on Flemming. For the
five-pipeline V2/V3 in-domain campaign use [UCloud release evaluation](ucloud-release.md);
for current release results and qualification use [the deployment freeze](deployment-freeze.md).

## Collect predictions

Run from the repository root in the existing environment without synchronization,
or in a separately prepared runtime following the [deployment install guide](../../../deployment/README.md).
Use a fresh output for each phase; inspect qualification before starting full runs.

```sh
python -m dev.releases.mambo_v3.evaluate prepare \
  --root /path/to/flemming \
  --reference /path/to/production/evaluation/expert/predictions/mini_metric.csv \
  --output /path/to/flemming-manifest.json
python -m dev.releases.mambo_v3.run_local qualification --precision auto \
  --python /path/to/runtime-env/bin/python \
  --bundle /path/to/bundle --manifest /path/to/flemming-manifest.json \
  --root /path/to/flemming --output /path/to/new-subset-run
```

Preparation joins all three archived truth ranks by original species/image identity,
hashes image bytes, and rejects missing/extra images or duplicate/incomplete truth.
It preserves out-of-vocabulary labels. Qualification selects 256 images from that
population; it does not redefine the full dataset or supplied training/test splits.

The historical runner defaults to `--precision fp32`; explicit `auto` matches
deployment defaults. TTA is off unless requested; pin a recipe when reproducing
historical studies. Replace `qualification` with `full` for two full GPU
prediction runs, or `benchmark` for isolated timing trials, changing the output
directory each time. Jobs are sequential; do not overlap timing with other work.

The collector checks hashes and applies full, legacy Europe/northern Europe and
updated European lists to each inference batch. Qualification also checks a
custom list equivalent to updated Europe, prediction/embedding agreement and
1280-dimensional unit embeddings. Embeddings are written incrementally to NPY.
Streaming controls and current ownership boundaries are described in the
[pipeline review](../../../docs/mambo-inference-pipeline-review.md).

Retained outputs include ordered sample identities, artifact/list hashes, precision,
runtime versions, canonical prediction CSVs and completion/failure reports. Use only
`complete` reports. Paired comparison checks identity/truth and label agreement;
all predictive metrics come from `mini_metrics`:

```sh
python -m dev.releases.mambo_v3.compare_quality /path/to/left /path/to/right \
  --output /path/to/comparison.json
```

## Metrics

Use the campaign's pinned metric environment, independently of runtime dependencies:

```sh
uv venv --python 3.13 /path/to/metrics-env
uv pip install --python /path/to/metrics-env/bin/python \
  'mini_metrics @ git+https://github.com/GuillaumeMougeot/mini_metrics.git@70cc69adc05362863439277048e06386c1f885e1'
/path/to/metrics-env/bin/python -m dev.releases.mambo_v3.metrics \
  --collection /path/to/full-run
```

For one list, use `--source /path/to/mini_metric.csv --output /path/to/metrics.json`
instead. Existing reports are reused only when input hash and metric revision match.
Schema `mini-metrics-quality-v2` rejects the older direct-accuracy extractor.

The report retains macro accuracy (`accuracy`), `micro_accuracy`, F1, recall,
precision, coverage and Theil U, plus per-class and known-only results at every rank.
Undefined values remain null. Truth-vocabulary coverage differs from abstention
coverage; the latter is 100% at this collector's zero threshold.
[Threshold calibration](../../../docs/mambo-confidence-thresholds.md) uses a
separate calibration partition, and [tail reporting](../../../docs/mambo-tail-metrics.md)
changes the macro averaging domain. Those later analyses reuse predictions.

## Timing and summary

`run_local benchmark` runs three alternating-order fresh-process trials per
PyTorch/ONNX × CPU/CUDA × predictions/embeddings setting. Four-thread CPU uses
batches 1/8; GPU adds 32; one-thread CPU adds batch-1 measurements. All cells use
the same seeded 32-image bank, two warmups and seven observations. Preserve raw
observations and trial ranges; this is a bounded sweep, not maximum throughput.

End-to-end timing covers decoding through completed CPU results, including
hierarchy reduction and requested embeddings. Prepared-runtime timing omits image
preparation and hierarchy reduction but includes transfers. Runtime import/setup,
predictor construction and lazy first call are separate; neither is cold-boot
latency. Nested startup components must not be summed. Reports retain resolved
precision, so FP32 reference runs and automatic-precision runs remain distinguishable.

RSS is the process high-water mark across the sweep. PyTorch allocator peaks are
not total VRAM; `nvidia-smi` snapshots are not continuous ONNX peaks. Record power,
clocks, temperature and unavailable policy fields. A CUDA provider can place some
operators on CPU; placement needs a separate profiler, whose timing is excluded.

```sh
python -m dev.releases.mambo_v3.summarize \
  --quality /path/to/full-run --benchmarks /path/to/benchmark-run \
  --output /path/to/new-summary
```

The summary rejects incomplete timing phases. Preserve its source CSVs, observations
and reports. [Initial FP32 results](../../../docs/mambo-v3-evaluation.md) retain the
original environment and installed-package checks; [accelerated results](../../../docs/mambo-accelerated-deployment.md)
record the later AMP comparison. In-domain collection and presentation have since
completed; cross-OS and additional hardware support must not be inferred from them.
