"""Compare same-stream H2D with one-batch lookahead around fixed ResNet compute."""

import json
import statistics
import time
from argparse import ArgumentParser

import torch
from torchvision.models import resnet18

from mini_trainer.data.io import LazyDataset
from mini_trainer.data.loader import get_dataloader


def run(samples=512, size=224, batch_size=32, repeats=5, backward=False, dtype="float16"):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA transfer benchmark requires an accessible CUDA device.")
    torch.manual_seed(42)
    device = torch.device("cuda:0")
    images = torch.randint(0, 256, (samples, 3, size, size), dtype=torch.uint8)
    dataset = LazyDataset(lambda item: images[item[0]], (list(range(samples)),), cache="cpu", cache_workers=0)
    direct = LazyDataset(
        lambda item: images[item[0]],
        (list(range(samples)),),
        cache="cpu",
        cache_workers=0,
        pin_batches=True,
    )
    variants = {
        "same_stream": (dataset, False),
        "prefetch": (dataset, True),
        "pinned_gather": (direct, False),
        "pinned_gather_prefetch": (direct, True),
    }
    loaders = {
        name: get_dataloader(data, "val", batch_size, 0, True, device, cuda_prefetch=prefetch)
        for name, (data, prefetch) in variants.items()
    }
    model = resnet18(weights=None).eval().to(device)
    precision = getattr(torch, dtype)
    timings = {name: [] for name in loaders}
    peaks = {name: [] for name in loaders}
    reference = None
    for trial in range(repeats + 1):
        for name in list(loaders) if trial % 2 else list(reversed(loaders)):
            model.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            outputs = []
            with torch.set_grad_enabled(backward):
                for batch in loaders[name]:
                    inputs = batch.to(device, non_blocking=True).float().div_(255)
                    with torch.autocast("cuda", dtype=precision, enabled=precision != torch.float32):
                        output = model(inputs)
                    outputs.append(output.detach())
                    if backward:
                        output.float().square().mean().backward()
                        model.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            peak = torch.cuda.max_memory_allocated(device)
            actual = torch.cat(outputs).cpu()
            if reference is None:
                reference = actual
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
            if trial:
                timings[name].append(elapsed)
                peaks[name].append(peak)
    return {
        "samples": samples,
        "size": size,
        "batch_size": batch_size,
        "workers": 0,
        "dtype": dtype,
        "backward": backward,
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "identical_outputs": True,
        "scope": (
            "CPU cache iteration, pinned H2D, scaling and fixed ResNet18 compute; "
            "BN frozen; no optimizer updates; excludes cache/model construction"
        ),
        "seconds": timings,
        "peak_cuda_allocated_bytes": peaks,
        "median_samples_per_second": {name: samples / statistics.median(values) for name, values in timings.items()},
        "speedup": statistics.median(timings["same_stream"]) / statistics.median(timings["prefetch"]),
        "speedups": {name: statistics.median(timings["same_stream"]) / statistics.median(values) for name, values in timings.items()},
    }


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--backward", action="store_true")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float16")
    args = parser.parse_args()
    if min(args.samples, args.size, args.batch_size, args.repeats) < 1:
        parser.error("Sizes and repeats must be positive")
    torch.set_num_threads(1)
    print(json.dumps(run(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
