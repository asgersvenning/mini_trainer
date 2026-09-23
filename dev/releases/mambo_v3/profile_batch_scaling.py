"""Causal batch-scaling probes; alternative layouts/precision are diagnostic only."""

import argparse
import contextlib
import cProfile
import pstats
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from deployment.mambo_deploy import Predictor
from dev.benchmarks.inference.onnx_inference import file_hash
from dev.releases.mambo_v3.benchmark import snapshot
from dev.releases.mambo_v3.evaluate import prepare_batch, runtime_settings
from dev.releases.mambo_v3.evaluation_data import load_records, write_json


def timed(call, repeats=7):
    values = []
    for _ in range(2):
        call()
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        call()
        torch.cuda.synchronize()
        values.append((time.perf_counter() - start) * 1000)
    return {"ms": values, "median_ms": statistics.median(values)}


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "runtime": runtime_settings(4), "before": snapshot(), "cells": [], "preparation": []}
    report.update(runner_sha256=file_hash(__file__), bundle_sha256=file_hash(args.bundle / "release.json"))
    write_json(args.output / "report.json", report)
    try:
        _, records = load_records(args.manifest, args.root, 32, 20260923)
        write_json(args.output / "samples.json", records)
        report["samples_sha256"] = file_hash(args.output / "samples.json")
        paths = [args.root / r["path"] for r in records]
        if any(file_hash(p) != r["sha256"] for p, r in zip(paths, records, strict=True)):
            raise ValueError("Image bytes changed")
        predictor = Predictor(args.bundle, backend="torch", device="cuda:0", model="north_europe", batch_size=32, threads=4)
        predictor.predict(paths[:1])
        model = predictor._torch_model
        report["model"] = str(model)
        arrays = {n: prepare_batch(paths[:n]) for n in (1, 8, 32)}
        tensors = {n: torch.from_numpy(x).cuda() for n, x in arrays.items()}
        # Establish model call sizes, not just the public API's requested batch.
        shapes = []
        hook = model.register_forward_pre_hook(lambda module, inputs: shapes.append(list(inputs[0].shape)))
        for n in arrays:
            predictor.predict(paths[:n])
        hook.remove()
        report["observed_forward_shapes"] = shapes
        with torch.inference_mode():
            for trial in range(3):
                modes = [(False, False), (True, False), (False, True), (True, True)]
                for amp, channels_last in modes if trial != 1 else reversed(modes):
                    layout = torch.channels_last if channels_last else torch.contiguous_format
                    model.to(memory_format=layout)
                    for n in (1, 8, 32) if trial != 1 else (32, 8, 1):
                        x = tensors[n].contiguous(memory_format=layout)

                        def compute():
                            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                                return model(x)

                        row = {"trial": trial, "batch": n, "amp": amp, "channels_last": channels_last}
                        row["resident_model"] = timed(compute)
                        row["hardware"] = snapshot()
                        report["cells"].append(row)
                        print(trial, n, amp, channels_last, row["resident_model"]["median_ms"], flush=True)
            model.to(memory_format=torch.contiguous_format)
            for n in (8, 32):
                with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], record_shapes=True
                ) as prof:
                    for _ in range(3):
                        model(tensors[n])
                    torch.cuda.synchronize()
                (args.output / f"torch-profile-{n}.txt").write_text(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=30))
                prof.export_chrome_trace(str(args.output / f"torch-profile-{n}.json"))
            for workers in (0, 4, 8):
                with ThreadPoolExecutor(max_workers=workers) if workers else contextlib.nullcontext(None) as pool:
                    for n in (1, 8, 32):

                        def prepare():
                            return prepare_batch(paths[:n], pool)

                        np.testing.assert_array_equal(prepare(), arrays[n])
                        row = {"workers": workers, "batch": n, "preparation": timed(prepare)}
                        # Intervention in the adapter only; original preprocessing values are unchanged.
                        with patch("deployment.mambo_deploy.predictor.preprocess") as mocked:

                            def prediction():
                                ready = iter(prepare())
                                mocked.side_effect = lambda item: next(ready)
                                return predictor.predict(paths[:n])

                            row["end_to_end"] = timed(prediction)
                        report["preparation"].append(row)
                        print("workers", workers, n, row["preparation"]["median_ms"], row["end_to_end"]["median_ms"], flush=True)
        prof = cProfile.Profile()
        prof.runcall(prepare_batch, paths)
        with (args.output / "cpu-profile.txt").open("w") as stream:
            pstats.Stats(prof, stream=stream).sort_stats("cumtime").print_stats(25)
        report["status"] = "complete"
    except Exception as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        report["after"] = snapshot()
        write_json(args.output / "report.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "manifest", "root", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    run(parser.parse_args())
