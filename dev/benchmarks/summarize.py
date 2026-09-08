"""Render retained JSON benchmark reports as a repository/Actions summary."""

import json
import statistics
from argparse import ArgumentParser
from pathlib import Path


def summarize(directory: Path) -> str:
    lines = [
        "# Dataset benchmark results",
        "",
        "| Run | Status | Device / precision | QT coverage | Accuracy by level | Parameter bytes | Peak CUDA MiB | "
        "Median train epoch 3+ | Training wall time |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    reports = sorted(directory.rglob("report.json"))
    for path in reports:
        report = json.loads(path.read_text())
        accuracy = ", ".join(f"{value:.2%}" for value in report.get("level_accuracies", [])) or "—"
        seconds = report.get("training_wall_seconds")
        duration = f"{seconds:.2f}s" if seconds is not None else "—"
        later_epochs = [
            phase["seconds"] for phase in report.get("phase_measurements", []) if phase["phase"] == "train" and phase["epoch"] >= 2
        ]
        later_duration = (
            f"{statistics.median(later_epochs):.3f}s"
            if later_epochs and report.get("phase_measurement_scope")
            else "unverified"
            if later_epochs
            else "—"
        )
        name = path.parent.relative_to(directory).as_posix()
        device = f"{report.get('device', '?')} / {report.get('dtype', '?')}"
        recipe = report.get("quantization_recipe")
        quantization = (
            f"INT8 ({len(recipe['quantized_modules'])} Linear)" if recipe else "requested" if report.get("quantized_training") else "off"
        )
        parameter_bytes = report.get("parameter_bytes", "—")
        peak = report.get("peak_cuda_allocated_bytes")
        peak_memory = (
            f"{peak / 2**20:.2f}"
            if peak is not None and report.get("peak_cuda_memory_scope")
            else "unverified"
            if peak is not None
            else "—"
        )
        lines.append(
            f"| {name} | {report['status']} | {device} | {quantization} | {accuracy} | {parameter_bytes} | "
            f"{peak_memory} | {later_duration} | {duration} |"
        )
    if not reports:
        lines.append("| No reports produced | incomplete | — | — | — | — | — | — | — |")
    lines.extend(
        [
            "",
            "Synthetic profiles require 100% oracle accuracy. Real-data runs marked `completed`",
            "have no quality acceptance threshold yet; completion does not establish an improvement.",
            "",
            "Wall times include setup, training, validation, logging and checkpoints. Compare timings",
            "only with matching hardware, dataset/configuration and timing scope. See JSON reports",
            "for provenance, errors and explicit coverage flags. CPU results do not validate GPU behavior.",
            "QT coverage counts quantized Linear modules; other operations may remain floating point.",
            "Parameter bytes describe stored parameters. CUDA peaks cover training, excluding final held-out inference.",
            "Older CUDA readings without a scope marker are unverified because logger resets could hide earlier peaks.",
            "Later-epoch medians use timed training phases from epoch 3 onward, including loading, preprocessing and batch logging.",
            "They exclude validation/figures/checkpoints, but may still include later compilation; they do not replace total wall time.",
        ]
    )
    return "\n".join(lines) + "\n"


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(summarize(args.directory), end="")


if __name__ == "__main__":
    main()
