"""Measure CPU cache construction from reproducible PNGs; exclude file creation.

An optional developer-supplied baseline io.py is executed as Python source to
compare an earlier implementation in the same dependency environment.
"""

import gc
import hashlib
import importlib.util
import json
import statistics
import sys
import tempfile
import time
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from mini_trainer.data._workers import _available_cpu_count
from mini_trainer.data.io import LazyDataset, make_read_and_resize_fn
from mini_trainer.data.loader import PathLabelProcessor


def run(samples=2048, size=128, repeats=5, workers=None, baseline_source=None):
    implementations = {"bounded": LazyDataset}
    if baseline_source is not None:
        name = "mini_trainer.data._benchmark_baseline"
        spec = importlib.util.spec_from_file_location(name, baseline_source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        implementations["baseline"] = module.LazyDataset
    reader = PathLabelProcessor(make_read_and_resize_fn((size, size), torch.device("cpu"), torch.uint8), None, False)
    timings = {name: [] for name in implementations}
    generator = np.random.default_rng(42)
    with tempfile.TemporaryDirectory(prefix="mini-trainer-cache-probe-") as root:
        paths = []
        # Repeat a fixed small corpus, measuring warm filesystem-cache decoding.
        for index in range(min(64, samples)):
            path = Path(root) / f"{index}.png"
            Image.fromarray(generator.integers(0, 256, (size, size, 3), dtype=np.uint8)).save(path)
            paths.append(str(path))
        items = ([paths[index % len(paths)] for index in range(samples)], list(range(samples)))
        expected_first = reader((items[0][0], 0))[0]
        for trial in range(repeats + 1):
            order = list(implementations) if trial % 2 else list(reversed(implementations))
            for name in order:
                gc.collect()
                options = {"cache_workers": workers} if name == "bounded" else {}
                started = time.perf_counter()
                dataset = implementations[name](reader, items, cache="cpu", **options)
                seconds = time.perf_counter() - started
                assert len(dataset) == samples
                torch.testing.assert_close(dataset[0][0], expected_first, rtol=0, atol=0)
                assert torch.equal(dataset._ram_cache.tensors[1], torch.arange(samples))
                if trial:
                    timings[name].append(seconds)
                del dataset
    return {
        "seed": 42,
        "samples": samples,
        "shape": [3, size, size],
        "threads": torch.get_num_threads(),
        "cache_workers": workers,
        "available_cpus": _available_cpu_count(),
        "source_sha256": {
            name: hashlib.sha256(Path(sys.modules[implementation.__module__].__file__).read_bytes()).hexdigest()
            for name, implementation in implementations.items()
        },
        "scope": "CPU cache construction from 64 repeated PNGs; warm filesystem cache; excludes file generation",
        "seconds": timings,
        "median_samples_per_second": {name: samples / statistics.median(values) for name, values in timings.items()},
        "speedup": statistics.median(timings["baseline"]) / statistics.median(timings["bounded"]) if baseline_source else None,
    }


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--baseline-source", type=Path)
    args = parser.parse_args()
    if min(args.samples, args.size, args.repeats) < 1 or (args.workers is not None and args.workers < 0):
        parser.error("Sizes and repeats must be positive; workers must be nonnegative")
    torch.set_num_threads(1)
    print(json.dumps(run(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
