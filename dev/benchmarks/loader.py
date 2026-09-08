"""Measure scalar versus batched cached loading without changing data or sampling."""

import json
import statistics
import time
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader, Dataset

from mini_trainer.data.io import LazyDataset
from mini_trainer.data.loader import get_dataloader


class ScalarFetch(Dataset):
    """The previous per-sample fetch contract, deliberately without __getitems__."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]


def run(samples=2048, size=64, batch_size=64, repeats=5, pin_batches=False):
    generator = torch.Generator().manual_seed(42)
    images = torch.randint(0, 256, (samples, 3, size, size), dtype=torch.uint8, generator=generator)
    dataset = LazyDataset(lambda item: (images[item[0]], torch.tensor(item[0])), (list(range(samples)),), cache="cpu")
    loaders = {
        "scalar": DataLoader(ScalarFetch(dataset), batch_size=batch_size, num_workers=0),
        "batched": get_dataloader(dataset, "val", batch_size, 0, False, torch.device("cpu")),
    }
    if pin_batches:
        if not torch.cuda.is_available():
            raise RuntimeError("Pinned batch comparison requires an accessible CUDA device.")
        direct = LazyDataset(
            lambda item: (images[item[0]], torch.tensor(item[0])),
            (list(range(samples)),),
            cache="cpu",
            cache_workers=0,
            pin_batches=True,
        )
        loaders = {
            "gather_then_pin": get_dataloader(dataset, "val", batch_size, 0, True, torch.device("cuda:0")),
            "pinned_gather": get_dataloader(direct, "val", batch_size, 0, True, torch.device("cuda:0")),
        }
    baseline, candidate = list(loaders)
    for expected, actual in zip(loaders[baseline], loaders[candidate], strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    timings = {name: [] for name in loaders}
    for trial in range(repeats + 1):
        # Alternate order to avoid consistently giving one variant warm caches.
        for name in list(loaders) if trial % 2 else list(reversed(loaders)):
            started = time.perf_counter()
            count = sum(len(batch[0]) for batch in loaders[name])
            elapsed = time.perf_counter() - started
            assert count == samples
            if trial:
                timings[name].append(elapsed)
    return {
        "samples": samples,
        "shape": [3, size, size],
        "dtype": "uint8",
        "batch_size": batch_size,
        "workers": 0,
        "threads": torch.get_num_threads(),
        "identical_batches": True,
        "pin_batches": pin_batches,
        "scope": "cached CPU loader iteration; excludes cache construction, preprocessing, H2D and model compute",
        "seconds": timings,
        "median_samples_per_second": {name: samples / statistics.median(values) for name, values in timings.items()},
        "speedup": statistics.median(timings[baseline]) / statistics.median(timings[candidate]),
    }


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--pin-batches", action="store_true")
    args = parser.parse_args()
    if min(args.samples, args.size, args.batch_size, args.repeats) < 1:
        parser.error("All sizes and repeat counts must be positive")
    torch.set_num_threads(1)
    print(json.dumps(run(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
