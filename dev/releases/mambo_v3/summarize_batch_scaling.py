"""Compact diagnostic evidence without private images or large profiler traces."""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from dev.benchmarks.inference.onnx_inference import file_hash
from dev.releases.mambo_v3.evaluation_data import write_json


def summarize(root, output):
    report = json.loads((root / "report.json").read_text())
    if report["status"] != "complete":
        raise ValueError("Incomplete batch diagnosis")
    grouped = defaultdict(list)
    for row in report["cells"]:
        grouped[row["amp"], row["channels_last"], row["batch"]].extend(row["resident_model"]["ms"])
    gpu = []
    for (amp, layout, batch), values in grouped.items():
        if len(values) != 21:
            raise ValueError("Expected three seven-observation trials")
        gpu.append(
            dict(
                amp=amp,
                channels_last=layout,
                batch=batch,
                median_ms=statistics.median(values),
                images_per_second=batch * 1000 / statistics.median(values),
            )
        )
    probes = json.loads((root / "preprocess-interventions.json").read_text())
    grouped = defaultdict(list)
    for row in probes:
        if not row["byte_identical_32"]:
            raise ValueError("Changed preprocessing values")
        grouped[row["variant"]].extend(row["ms"])
    cpu = [
        {"variant": name, "batch": 32, "median_ms": statistics.median(values), "observations": len(values)}
        for name, values in grouped.items()
    ]
    events = json.loads((root / "torch-profile-32.json").read_text())["traceEvents"]
    kernels = [e for e in events if e.get("cat") == "kernel"]
    kernel_us = sum(e["dur"] for e in kernels)
    predicates = {
        "convolution": lambda name: any(part in name for part in ("scudnn", "convolve", "conv_depthwise")),
        "batch_normalization": lambda name: "bn_fw_inf" in name,
        "silu": lambda name: "silu_kernel" in name,
    }
    shares = {name: sum(e["dur"] for e in kernels if predicate(e["name"])) / kernel_us for name, predicate in predicates.items()}
    source_names = (
        "report.json",
        "preprocess-interventions.json",
        "preprocess-line-profile.json",
        "preprocess-layout.json",
        "torch-profile-32.json",
        "onnx-profile-summary.json",
    )
    write_json(
        output,
        {
            "status": "complete",
            "hardware": report["before"],
            "observed_forward_shapes": report["observed_forward_shapes"],
            "gpu_resident": gpu,
            "preparation_interventions": cpu,
            "threaded_preparation": report["preparation"],
            "preprocess_layout": json.loads((root / "preprocess-layout.json").read_text()),
            "preprocess_line_profile": json.loads((root / "preprocess-line-profile.json").read_text()),
            "cuda_kernel_time_shares_batch32": shares,
            "onnx_placement": json.loads((root / "onnx-profile-summary.json").read_text()),
            "source_sha256": {name: file_hash(root / name) for name in source_names},
            "limits": (
                "Diagnostic interventions, not qualified release variants. GPU timings exclude input preparation and transfers; "
                "threaded end-to-end uses unchanged pixels. Profiler timings are explanatory, not replacement benchmarks. "
                "No hardware counter roofline attribution."
            ),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.evidence, args.output)
