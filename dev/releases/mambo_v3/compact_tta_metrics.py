"""Evaluate compact TTA qualification predictions using pinned mini_metrics."""

import argparse
import csv
import json
from pathlib import Path

from dev.releases.mambo_v3.evaluation_data import write_json
from dev.releases.mambo_v3.metrics import REVISION, finite_json, require_pinned_metrics


def collect(root, study):
    from mini_metrics.data import MetricDF
    from mini_metrics.metrics import evaluate_file

    require_pinned_metrics()
    report = json.loads((root / "report.json").read_text())
    if report["status"] != "complete":
        raise ValueError("Incomplete inference")
    thresholds = study["models"]["torch"]["thresholds"]
    scores, transitions = {}, {}
    for name in report["recipes"]:
        scores[name], transitions[name] = {}, {}
        for subset in report["splits"]:
            path = root / name / f"{subset}.csv"
            data = MetricDF.from_source(path)
            scores[name][subset] = {
                scope: finite_json(
                    evaluate_file(
                        data,
                        threshold=threshold,
                        simple=True,
                        hierarchical=False,
                        pattern=r"^(accuracy|micro_accuracy|precision|recall|f1|coverage)$",
                        verbose=0,
                    )
                )
                for scope, threshold in (("zero", 0), ("fixed", thresholds))
            }
            with (root / "none" / f"{subset}.csv").open() as stream:
                baseline = list(csv.DictReader(stream))
            with path.open() as stream:
                candidate = list(csv.DictReader(stream))
            transitions[name][subset] = {}
            for level, rank in enumerate(("species", "genus", "family")):
                counts = dict.fromkeys(
                    (
                        "correct",
                        "accepted_wrong",
                        "wrong_to_correct",
                        "correct_to_wrong",
                        "wrong_accepted_to_rejected",
                        "correct_accepted_to_rejected",
                    ),
                    0,
                )
                for a, b in zip(baseline, candidate, strict=True):
                    if a["level"] != str(level):
                        continue
                    assert (a["instance_id"], a["label"], a["level"]) == (b["instance_id"], b["label"], b["level"])
                    ac, bc = a["prediction"] == a["label"], b["prediction"] == b["label"]
                    aa, ba = float(a["confidence"]) >= thresholds[level], float(b["confidence"]) >= thresholds[level]
                    for key, test in (
                        ("correct", bc),
                        ("accepted_wrong", not bc and ba),
                        ("wrong_to_correct", not ac and bc),
                        ("correct_to_wrong", ac and not bc),
                        ("wrong_accepted_to_rejected", not ac and aa and not ba),
                        ("correct_accepted_to_rejected", ac and aa and not ba),
                    ):
                        counts[key] += int(test)
                transitions[name][subset][rank] = counts
        print(name, flush=True)
    return {"revision": REVISION, "fixed_thresholds": thresholds, "scores": scores, "transitions": transitions}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--study", type=Path, default=Path("docs/assets/mambo-threshold-comparison.json"))
    args = parser.parse_args()
    write_json(args.root / "metrics.json", collect(args.root, json.loads(args.study.read_text())))
