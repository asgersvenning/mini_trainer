"""Flemming trap-crop evaluation in lepinet's format: trap, probe and probe-HO from one predict run.

    python dev/lepinet/flemming.py index --parquet <flemming_names_labels.parquet> --images <flemming/images> --out <index.json>
    mt_hpredict --weights <last.pt> --input <flemming/images> --data_index <index.json> --output <dir> --name trap
    python dev/lepinet/flemming.py score --predictions <dir>/trap/mini_metric.csv --lepinet <lepinet checkout>

``score`` writes ``<set>/predictions.csv`` (lepinet's layout, filename ``<species folder>/<crop>``) and
``<set>/scores.csv`` next to its input. Scores are lepinet's ``dev/093_rescore_mini_metrics.py``: zero and
random calibration (``score``) and calibration on whole (trap, night) groups (``score_grouped``); probe-HO keeps
the held-out species only; ``--sets adapt`` adds the adapt nights (for choosing settings there before reporting
on probe and probe-HO); ``--set-file NAME=PARQUET`` scores any other crop list (``image_path`` column, e.g. a half
of the probe nights; a NAME ending in ``-ho`` keeps the held-out species only, as probe-HO does). Run ``score``
with pandas, scikit-learn and mini_metrics on the path, e.g. lepinet's venv with ``PYTHONPATH=<lepinet>/data/tools/mini_metrics-61db373``.
"""

import argparse
import importlib.util
import json
from pathlib import Path

import pandas as pd

COLUMNS = ["instance_id", "filename", "level", "label", "prediction", "confidence", "threshold", "known_label"]
SETS = {
    "trap": None,
    "adapt": "flemming_adapt.parquet",
    "probe": "flemming_probe.parquet",
    "probe-ho": "flemming_probe_heldout_species.parquet",
}
DEFAULT_SETS = ("trap", "probe", "probe-ho")


def index(args):
    rows = pd.read_parquet(args.parquet, columns=["image_path", "speciesKey", "genusKey", "familyKey"])
    keys = rows[["speciesKey", "genusKey", "familyKey"]].apply(  # GBIF keys are numeric; Catalogue of Life keys are strings
        lambda column: (column.astype("int64") if pd.api.types.is_numeric_dtype(column) else column).astype(str)
    )
    data = {
        "path": [str(args.images.resolve() / p) for p in rows.image_path],
        "label": keys.values.tolist(),
        "class": [0] * len(rows),  # unused by predict; set so out-of-vocabulary labels need no class index
        "split": ["test"] * len(rows),
    }
    args.out.write_text(json.dumps(data))
    print(f"{len(rows):,} images, {rows.speciesKey.nunique()} species -> {args.out}")


def score(args):
    spec = importlib.util.spec_from_file_location("rescore", args.lepinet / "dev" / "093_rescore_mini_metrics.py")
    rescore = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rescore)
    import mini_metrics  # noqa: F401  (fail early if it is not on the path)

    splits = args.lepinet / "data" / "flemming" / "splits"
    heldout = {str(s) for s in json.loads((splits / "split_summary.json").read_text())["held_out_species"]}
    rows = pd.read_csv(args.predictions, dtype={"label": str, "prediction": str})
    rows["filename"] = rows.filename.str.split("/").str[-2:].str.join("/")
    rows["known_label"] = rows.known_label.astype(str).isin(["1", "True", "true"])
    files = {name: splits / SETS[name] if SETS[name] else None for name in args.sets}
    for item in args.set_file:
        name, _, path = item.partition("=")
        if not name or not path or name in files:
            raise SystemExit(f"--set-file wants a new NAME=PARQUET, got {item!r}")
        files[name] = Path(path)
    for name, split in files.items():
        keep = rows if split is None else rows[rows.filename.isin(set(pd.read_parquet(split).image_path))]
        out = args.predictions.parent / name / "predictions.csv"
        out.parent.mkdir(exist_ok=True)
        keep[COLUMNS].to_csv(out, index=False)
        ho = heldout if name.endswith("-ho") else None
        grouped = pd.DataFrame(rescore.score_grouped(out, ho, args.seeds, splits / "flemming_groups.parquet"))
        plain = pd.DataFrame(rescore.score(out, ho, args.seeds, None))
        scores = plain.merge(grouped[["level", "metric", "n_groups", "calg_mean", "calg_sd"]], on=["level", "metric"])
        scores.to_csv(out.with_name("scores.csv"), index=False)
        f1 = scores[scores.metric == "f1"].set_index("level")
        print(
            f"{name}: {keep.instance_id.nunique():,} images, {int(f1.n_groups.iloc[0])} groups; macro-F1 by level "
            f"zero {f1.zero.round(4).tolist()}, (trap, night)-calibrated {f1.calg_mean.round(4).tolist()}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(required=True)
    p = sub.add_parser("index", help="MT data index for lepinet's trap parquet.")
    p.add_argument("--parquet", type=Path, required=True)
    p.add_argument("--images", type=Path, required=True, help="Folder of <species folder>/<crop> images.")
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(run=index)
    p = sub.add_parser("score", help="Split, convert and score one predict output.")
    p.add_argument("--predictions", type=Path, required=True, help="mini_metric.csv from mt_hpredict on the index.")
    p.add_argument("--lepinet", type=Path, required=True, help="lepinet checkout (dev/093 and data/flemming/splits).")
    p.add_argument("--seeds", type=int, default=20, help="Calibration splits (dev/093's default).")
    p.add_argument("--sets", nargs="+", choices=list(SETS), default=list(DEFAULT_SETS), help="Crop sets to score.")
    p.add_argument("--set-file", action="append", default=[], metavar="NAME=PARQUET", help="Another crop set (repeatable).")
    p.set_defaults(run=score)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
