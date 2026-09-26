"""Small pipeline diagnostic with synthetic device scores in place of model execution."""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from itertools import batched
from pathlib import Path
from unittest.mock import patch

import torch

from deployment.mambo_deploy import Predictor
from deployment.mambo_deploy import predictor as predictor_module
from deployment.mambo_deploy.preprocessing import prepare_batch
from deployment.mambo_deploy.result_worker import ResultWorker
from deployment.mambo_deploy.transfers import device_batches, pinned_factory
from dev.benchmarks.inference.onnx_inference import file_hash
from dev.releases.mambo_v3.evaluation_data import load_records, write_json


@contextmanager
def measured_results(predictor, stats):
    class MeasuredWorker(ResultWorker):
        def __init__(self, function):
            def process(*args):
                start = time.perf_counter()
                try:
                    return function(*args)
                finally:
                    stats["result_worker_seconds"] += time.perf_counter() - start

            super().__init__(process)

        def pop(self):
            start = time.perf_counter()
            try:
                return super().pop()
            finally:
                stats["result_wait_seconds"] += time.perf_counter() - start

    ranked = predictor._ranked_views

    def submit(*args, **kwargs):
        start = time.perf_counter()
        try:
            return ranked(*args, **kwargs)
        finally:
            stats["submission_seconds"] += time.perf_counter() - start

    with patch.object(predictor_module, "ResultWorker", MeasuredWorker), patch.object(predictor, "_ranked_views", submit):
        yield


def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to retain the deployed transfer and preprocessing path")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    _, records = load_records(args.manifest, args.root, args.count)
    paths = [args.root / record["path"] for record in records]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        # Warm the selected files once, outside timings; retain their identity.
        hashes = list(pool.map(file_hash, paths))
    p = Predictor(args.bundle, backend="torch", device="cuda:0", model="full", batch_size=args.batch_size, precision="auto")
    plan = p.hierarchy_plan(p.selected)
    generator = torch.Generator(device=p.device).manual_seed(20260925)
    leaf = torch.randn((args.batch_size, len(plan.labels[0])), generator=generator, device=p.device)
    scores = plan.torch_values(leaf)[0]

    def mock_model(images, embeddings, *, tensors=False):
        assert tensors and not embeddings
        p._device_preprocess(images)  # Keep the actual batched uint8 -> FP32 preprocessing.
        return [score[: len(images)] for score in scores], None

    # One real prepared batch is reused only in the modes that exclude input preparation.
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        prepared = prepare_batch(paths[: args.batch_size], pool, compact=True, decode=p._decode)
    host = pinned_factory(p.device, compact=True)(prepared.shape)
    host[...] = prepared
    resident = torch.from_numpy(host).to(p.device)
    report = {
        "boundary": "Synthetic native rank logits replace backbone/head; GPU preprocessing, transfers and Prediction remain real.",
        "limitations": "Zero model latency changes overlap/backpressure. Local diagnostic rates are not HPC deployment estimates.",
        "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "torch": torch.__version__,
        "gpu": str(torch.cuda.get_device_properties(p.device)),
        "samples": [{"path": str(path), "sha256": digest} for path, digest in zip(paths, hashes, strict=True)],
        "cells": [],
    }
    with patch.object(p, "_torch", mock_model):
        for mode in ("resident", "host", "stream"):

            def prepared_batches(items, batch_size, *, stats, **options):
                def source():
                    for index, batch in enumerate(batched(items, batch_size)):
                        yield index * batch_size, (host[: len(batch)],)

                if mode == "host":
                    yield from device_batches(source(), "torch", p.device, stats)
                else:
                    for offset, views in source():
                        count = len(views[0])
                        yield offset, (resident[:count],), count

            def consume(selected, stats):
                return sum(
                    len(result)
                    for result in p.predict_stream(
                        selected,
                        prepare_workers=args.workers,
                        read_workers=32,
                        read_window=max(args.batch_size, 128),
                        stats=stats,
                    )
                )

            replacement = p.prepared_batches if mode == "stream" else prepared_batches
            with patch.object(p, "prepared_batches", replacement):
                # Warm kernels/result paths once; warmup is excluded from all counters.
                consume(paths[: args.batch_size], {})
                torch.cuda.synchronize()
                stats = dict(result_worker_seconds=0.0, result_wait_seconds=0.0, submission_seconds=0.0)
                p.runtime_timings.clear()
                with measured_results(p, stats):
                    start = time.perf_counter()
                    count = consume(paths, stats)
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - start
                assert count == len(paths) and p._torch_model is None
                cell = dict(
                    mode=mode,
                    images=count,
                    seconds=elapsed,
                    images_per_second=count / elapsed,
                    pipeline=stats,
                    runtime=dict(p.runtime_timings),
                )
                report["cells"].append(cell)
                write_json(args.output / "report.json", report)
                print(json.dumps(cell), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "manifest", "root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--count", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if min(args.count, args.batch_size, args.workers) < 1 or args.count < args.batch_size:
        parser.error("Require count >= batch-size > 0 and workers > 0")
    run(args)


if __name__ == "__main__":
    main()
