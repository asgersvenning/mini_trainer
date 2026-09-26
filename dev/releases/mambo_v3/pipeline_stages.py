"""Bounded stage probe: synthetic images/scores, no filesystem latency or model execution."""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from deployment.mambo_deploy.preprocessing import TorchPreprocess, preprocess
from deployment.mambo_deploy.results import Prediction
from deployment.mambo_deploy.transfers import download_tensors


def timed(function, count):
    function()
    start = time.perf_counter()
    for _ in range(count):
        function()
    return (time.perf_counter() - start) / count


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(91025)
    rng = np.random.default_rng(91025)
    report = {
        "boundary": "Synthetic decoded RGB images and rank logits; no storage, decoding, backbone or classifier.",
        "limitations": "Stage timings exclude pipeline contention. Compare operation counts/storage as well as local elapsed time.",
        "settings": {"seed": 91025, "workers": 4, "rank_sizes": [30000, 4000, 500]},
        "numpy": np.__version__,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
    }
    for size in (256, 2048):
        image = rng.integers(0, 256, (size, size, 3), dtype=np.uint8).transpose(2, 0, 1)
        with ThreadPoolExecutor(4) as pool:
            report[f"cpu_prepare_{size}_seconds_per_32"] = timed(lambda: list(pool.map(preprocess, [image] * 32)), 3)
    for batch in (64, 256):
        raw = [rng.standard_normal((batch, n), dtype=np.float32) for n in report["settings"]["rank_sizes"]]
        labels = [[str(i) for i in range(v.shape[1])] for v in raw]
        mappings = [np.arange(v.shape[1]) for v in raw]
        report[f"results_{batch}_seconds"] = timed(lambda: Prediction(raw, labels, mappings), 5)

    preparation = TorchPreprocess(torch, "cuda:0")
    images = torch.randint(0, 256, (64, 3, 384, 384), dtype=torch.uint8, device="cuda:0")
    values = [torch.randn((64, n), device="cuda:0") for n in report["settings"]["rank_sizes"]]
    stream = torch.cuda.Stream()
    for _ in range(3):
        preparation(images)
        download_tensors(values, torch=torch, stream=stream)
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        profile_memory=True,
        record_shapes=True,
    ) as prof:
        with torch.profiler.record_function("GPU preprocessing"):
            prepared = preparation(images)
        with torch.profiler.record_function("Output transfer submission"):
            finish = download_tensors(values, torch=torch, stream=stream, defer=True)
        with torch.profiler.record_function("Output transfer completion"):
            finish()
        torch.cuda.synchronize()
    prof.export_chrome_trace(str(output / "trace.json"))
    (output / "operators.txt").write_text(prof.key_averages().table(sort_by="self_device_time_total", row_limit=40))
    report["prepared"] = {"shape": list(prepared.shape), "stride": list(prepared.stride()), "contiguous": prepared.is_contiguous()}
    report["gpu_operators"] = [
        {"name": event.key, "calls": event.count, "self_device_us": event.self_device_time_total, "device_bytes": event.device_memory_usage}
        for event in prof.key_averages()
        if event.key.startswith("aten::")
    ]
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
