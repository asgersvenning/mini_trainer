"""Audit predicted-only family contributions using pinned mini_metrics group outputs."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from dev.benchmarks.inference.onnx_inference import file_hash
from dev.releases.mambo_v3.evaluation_data import write_json
from dev.releases.mambo_v3.metrics import REVISION, finite_json, require_pinned_metrics
from dev.releases.mambo_v3.threshold_report import identity


def prepare_taxonomy(metadata, v2_path, v3_path):
    """Read names and verify hierarchy identity without constructing a model."""
    import pyarrow.parquet as pq
    import torch

    names = {}
    for batch in pq.ParquetFile(metadata).iter_batches(columns=["familyKey", "family"], batch_size=500000):
        columns = batch.to_pydict()
        for key, name in zip(columns["familyKey"], columns["family"], strict=True):
            if key is not None and name is not None:
                key = str(int(key))
                if key in names and names[key] != name:
                    raise ValueError(f"Conflicting family name: {key}")
                names[key] = name
    state = torch.load(v2_path, map_location="cpu", weights_only=True)
    classes = json.loads(v3_path.read_text())
    inverse = {k: {v: n for n, v in mapping.items()} for k, mapping in state["classifier._extra_state"]["cls2idx"].items()}
    old = {
        inverse["0"][i]: (inverse["1"][int(g)], inverse["2"][int(state["classifier.mask_1"][g])])
        for i, g in enumerate(state["classifier.mask_0"])
    }
    new = {
        n: (classes["labels"][1][g], classes["labels"][2][classes["parents"][1][g]])
        for n, g in zip(classes["labels"][0], classes["parents"][0], strict=True)
    }
    if old != new:
        raise ValueError("V2/V3 taxonomy mappings differ")
    active = [inverse["0"][i] for i in state["classifier.active_indices"].tolist()]
    return {
        "names": names,
        "sources": {str(p): file_hash(p) for p in (metadata, v2_path, v3_path)},
        "taxonomy": {
            "all_species_parent_mappings_identical": True,
            "species": len(old),
            "global_families": len(classes["labels"][2]),
            "north_europe_families": sorted({old[n][1] for n in active}),
        },
    }


def collect(study_path, taxonomy_path):
    from mini_metrics.data import MetricDF
    from mini_metrics.metrics import MacroF1, MacroPrecision, MacroRecall, evaluate_file

    require_pinned_metrics()
    study = json.loads(study_path.read_text())
    taxonomy = json.loads(taxonomy_path.read_text())
    result = {"revision": REVISION, "study_sha256": file_hash(study_path), "taxonomy": taxonomy, "models": {}}
    for model, source in study["models"].items():
        if file_hash(source["source"]) != source["source_sha256"]:
            raise ValueError("Changed prediction source")
        reporting, _ = MetricDF.from_source(source["source"]).split((0.9, 0.1), strata=("label",), seed=42)
        if identity(reporting) != source["identities"]["report"]:
            raise ValueError("Changed reporting partition")
        family = reporting[reporting.level == 2]
        rows = {}
        for scope, threshold in (("zero", 0), ("optimized", source["thresholds"][2]), ("common_0.96", 0.96)):
            data = family.with_threshold(threshold)
            truth = Counter(map(str, data.label))
            accepted = np.asarray(data.prediction_made)
            predicted = Counter(map(str, data.prediction[accepted]))
            groups = {}
            diagnostics = {}
            for name, metric in (("precision", MacroPrecision()), ("recall", MacroRecall()), ("f1", MacroF1())):
                values = metric(data, aggregate=False, verbose=0)[2]
                groups[name] = finite_json(values)
                # Reuse the package's aggregation, changing only which class groups enter it.
                diagnostics[name] = float(metric._aggregate_groups({k: v for k, v in values.items() if str(k) in truth}))
            official = finite_json(
                evaluate_file(data, simple=True, hierarchical=False, pattern=r"^(precision|recall|f1|coverage|micro_accuracy)$", verbose=0)
            )
            if scope != "common_0.96":
                ref = source["report_zero" if scope == "zero" else "report_optimized"]
                for metric, levels in official.items():
                    if not np.isclose(levels["2"], ref[metric]["2"], atol=1e-12, rtol=0):
                        raise ValueError("Official metric changed")
            pairs = Counter(zip(map(str, data.label[accepted]), map(str, data.prediction[accepted]), strict=True))
            rows[scope] = {
                "threshold": threshold,
                "images": len(data),
                "accepted": int(accepted.sum()),
                "truth_counts": dict(truth),
                "predicted_counts": dict(predicted),
                "official": official,
                "truth_group_diagnostic": diagnostics,
                "groups": groups,
                "predicted_only": {k: v for k, v in predicted.items() if k not in truth},
                "truth_families_without_accepted_predictions": sorted(set(truth) - set(predicted)),
                "false_positive_pairs": [{"truth": t, "prediction": p, "images": n} for (t, p), n in pairs.most_common() if t != p],
            }
        result["models"][model] = rows
    return result


def export(data, output):
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "mambo-family-precision.json", data)
    names = data["taxonomy"]["names"]
    with (output / "mambo-family-precision.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            [
                "model",
                "scope",
                "threshold",
                "family_id",
                "family",
                "truth_images",
                "accepted_predictions",
                "precision",
                "precision_weight",
                "recall",
                "recall_weight",
                "f1",
                "f1_weight",
            ]
        )
        for model, scopes in data["models"].items():
            for scope, row in scopes.items():
                for family in sorted(row["groups"]["f1"]):
                    writer.writerow(
                        [
                            model,
                            scope,
                            row["threshold"],
                            family,
                            names.get(family, family),
                            row["truth_counts"].get(family, 0),
                            row["predicted_counts"].get(family, 0),
                            *[v for metric in ("precision", "recall", "f1") for v in row["groups"][metric].get(family, (None, 0))],
                        ]
                    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=Path("docs/assets/mambo-threshold-comparison.json"))
    parser.add_argument("--taxonomy", type=Path, required=True, help="Verified family names and V2/V3 taxonomy provenance JSON")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export(collect(args.study, args.taxonomy), args.output)
