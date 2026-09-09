"""Compare streaming image readers with exact batches and file provenance."""

import hashlib
import json
import statistics
import time
from argparse import ArgumentParser
from importlib.metadata import version
from pathlib import Path

import torch
from torchvision.io import ImageReadMode, decode_image
from torchvision.transforms import functional as transforms

from mini_trainer.data.io import ReadAndResize
from mini_trainer.data.loader import get_inference_dataloader


class TorchvisionReader(ReadAndResize):
    """The former decode/float-resize/convert path, retained as a reference."""

    def __call__(self, path):
        if not isinstance(path, str):
            path = path[0]
        image = decode_image(path, mode=ImageReadMode.RGB, apply_exif_orientation=False)
        image = transforms.resize(image, [self.h, self.w], interpolation=self.interp, antialias=self.antialias)
        return (image if image.dtype == self.dtype else self.converter(image)).to(self.device)


def run(data_root, samples=128, size=224, batch_size=16, repeats=7, workers=0):
    root = Path(data_root).resolve()
    paths = sorted(path for path in root.rglob("*") if path.suffix.lower() in (".jpg", ".jpeg", ".png") and path.is_file())[:samples]
    if not paths:
        raise ValueError("No JPEG/PNG images found under the supplied data root.")
    manifest = [{"path": str(path.relative_to(root)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in paths]
    loaders = {}
    for name in ("torchvision", "gather"):
        dataset, loaders[name] = get_inference_dataloader(
            list(map(str, paths)),
            resize_size=size,
            batch_size=batch_size,
            num_workers=workers,
            multiprocessing_context="spawn" if workers else None,
        )
        if name == "torchvision":
            dataset.func = TorchvisionReader((size, size), torch.device("cpu"), torch.uint8)
    for expected, actual in zip(loaders["torchvision"], loaders["gather"], strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    timings = {name: [] for name in loaders}
    for trial in range(repeats + 1):
        for name in list(loaders) if trial % 2 else list(reversed(loaders)):
            started = time.perf_counter()
            count = sum(len(batch) for batch in loaders[name])
            elapsed = time.perf_counter() - started
            assert count == len(paths)
            if trial:
                timings[name].append(elapsed)
    return {
        "files": manifest,
        "samples": len(paths),
        "shape": [3, size, size],
        "dtype": "uint8",
        "batch_size": batch_size,
        "workers": workers,
        "threads": torch.get_num_threads(),
        "versions": {name: version(name) for name in ("torch", "torchvision", "numpy")},
        "identical_batches": True,
        "scope": (
            "uncached image loading, decoding, nearest resize, batch assembly and IPC with a warm filesystem cache; "
            "excludes worker startup, H2D and model compute"
        ),
        "seconds": timings,
        "median_samples_per_second": {name: len(paths) / statistics.median(values) for name, values in timings.items()},
        "speedup": statistics.median(timings["torchvision"]) / statistics.median(timings["gather"]),
    }


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()
    if min(args.samples, args.size, args.batch_size, args.repeats) < 1 or args.workers < 0:
        parser.error("Sizes/repeats must be positive and workers nonnegative.")
    torch.set_num_threads(1)
    print(json.dumps(run(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
