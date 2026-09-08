"""Render retained JSON benchmark reports as a repository/Actions summary."""

import json
from argparse import ArgumentParser
from pathlib import Path


def summarize(directory: Path) -> str:
    lines = [
        "# Dataset benchmark results",
        "",
        "| Run | Status | Device / precision | Accuracy by level | Training wall time |",
        "| --- | --- | --- | --- | --- |",
    ]
    reports = sorted(directory.rglob("report.json"))
    for path in reports:
        report = json.loads(path.read_text())
        accuracy = ", ".join(f"{value:.2%}" for value in report.get("level_accuracies", [])) or "—"
        seconds = report.get("training_wall_seconds")
        duration = f"{seconds:.2f}s" if seconds is not None else "—"
        name = path.parent.relative_to(directory).as_posix()
        device = f"{report.get('device', '?')} / {report.get('dtype', '?')}"
        lines.append(f"| {name} | {report['status']} | {device} | {accuracy} | {duration} |")
    if not reports:
        lines.append("| No reports produced | incomplete | — | — | — |")
    lines.extend(
        [
            "",
            "Synthetic profiles require 100% oracle accuracy. Real-data runs marked `completed`",
            "have no quality acceptance threshold yet; completion does not establish an improvement.",
            "",
            "Wall times include setup, training, validation, logging and checkpoints. Compare timings",
            "only with matching hardware, dataset/configuration and timing scope. See JSON reports",
            "for provenance, errors and explicit coverage flags. CPU results do not validate GPU behavior.",
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
