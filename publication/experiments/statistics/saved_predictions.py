"""Recompute unthresholded leaf recall from archived mini_metrics input CSVs."""

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from publication.experiments.artifacts import fingerprint


def summarize(path, path_marker="", expected=None):
    support, correct, confusion = Counter(), Counter(), Counter()
    identities = {}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["level"] != "0":
                continue
            if float(row["threshold"]) != 0 or row["prediction_made"] != "1" or row["known_label"] != "1":
                raise ValueError("Raw recall requires known labels and unthresholded predictions")
            name = row["filename"]
            if path_marker:
                if path_marker not in name:
                    raise ValueError(f"Missing sample path marker: {name}")
                name = path_marker + name.rsplit(path_marker, 1)[1]
            if name in identities:
                raise ValueError(f"Duplicate sample: {name}")
            target, predicted = row["label"], row["prediction"]
            identities[name] = target
            support[target] += 1
            correct[target] += target == predicted
            confusion[target, predicted] += 1
    if not identities:
        raise ValueError("No leaf predictions")
    if expected is not None and identities != expected:
        raise ValueError("Prediction sample identities or labels differ from the requested split")
    identity_hash = hashlib.sha256(json.dumps(sorted(identities.items()), separators=(",", ":")).encode()).hexdigest()
    metrics = {
        "samples": len(identities),
        "species": len(support),
        "macro_recall": sum(correct[k] / n for k, n in support.items()) / len(support),
        "micro_recall": sum(correct.values()) / len(identities),
        "sample_labels_sha256": identity_hash,
    }
    classes = [{"label": k, "support": n, "correct": correct[k], "recall": correct[k] / n} for k, n in sorted(support.items())]
    pairs = [{"label": k, "prediction": p, "count": n} for (k, p), n in sorted(confusion.items())]
    return metrics, classes, pairs


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Campaign directories containing results/runs/*/predict/<evaluation>/mini_metric.csv")
    parser.add_argument("output", type=Path)
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--path-marker", default="")
    parser.add_argument("--data-index", type=Path)
    parser.add_argument("--split", choices=["train", "validation", "test"], default="test")
    args = parser.parse_args()
    expected = None
    provenance = {"metric": "unthresholded level-0 recall", "split": args.split if args.data_index else "unverified", "inputs": {}}
    if args.data_index:
        index = json.loads(args.data_index.read_text())
        expected = {
            p: label[0] for p, label, split in zip(index["path"], index["label"], index["split"], strict=True) if split == args.split
        }
        provenance["data_index"] = fingerprint(args.data_index)
    summaries = []
    args.output.mkdir(parents=True, exist_ok=True)
    for path in sorted(args.root.glob(f"*/results/runs/*/predict/{args.evaluation}/mini_metric.csv")):
        relative = path.relative_to(args.root)
        campaign, run = relative.parts[0], relative.parts[3]
        metrics, classes, pairs = summarize(path, args.path_marker, expected)
        summaries.append({"campaign": campaign, "run": run, **metrics})
        provenance["inputs"][relative.as_posix()] = fingerprint(path)
        output = args.output / campaign / run
        output.mkdir(parents=True, exist_ok=True)
        write_csv(output / "classes.csv", classes)
        write_csv(output / "confusion.csv", pairs)
    if not summaries:
        raise ValueError("No matching prediction files")
    write_csv(args.output / "summary.csv", summaries)
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
