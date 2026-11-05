"""Self-training data for the trap-shift runs: lepinet's dev/065 (gate), dev/066 (dose) and dev/089 (trap folds).

    python dev/lepinet/pseudo.py select --predictions <dir>/trap/mini_metric.csv --adapt <flemming_adapt.parquet> --out pseudo.parquet
    python dev/lepinet/pseudo.py merge --global-parquet <global.parquet> --labels pseudo.parquet --out <data>/train.parquet
    python dev/lepinet/pseudo.py pool --pattern '<preds>/holdout-{fold}/trap/mini_metric.csv' --folds trap --out <pooled.csv>

``select`` keeps the adapt crops whose species confidence reaches its ``1 - keep_frac`` quantile (ties kept), labelled by
the teacher or, with ``--true-labels``, by the trap data. ``merge`` appends them as set '2' to the global rows: species
with ``--min-count`` real rows outside set 0 only, coarse keys from the global taxonomy, replicated to ``--target-frac``
of all rows. It writes ``<out>.summary.json`` and ``<out>.staging.tsv`` (copy ``<flemming>/images/<src>`` to
``<dir of out>/images/<dst>``). ``pool`` keeps each crop's rows from the fold that never saw its trap or site.
"""

import argparse
import csv
import json
import re
from itertools import compress
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pv
import pyarrow.parquet as pq

KEYS = ("speciesKey", "genusKey", "familyKey", "orderKey", "classKey", "phylumKey", "kingdomKey")
TRAPS = [f"{s}{i}" for s in ("LV", "OH", "SS") for i in range(1, 5)]
FOLDS = {t: [t] for t in TRAPS} | {f"site-{s}": [f"{s}{i}" for i in range(1, 5)] for s in ("LV", "OH", "SS")}
TRAP_RE = re.compile(r"TRAPNAME_([^_]+)_")


def crop(path):
    """``<species folder>/<crop>`` from any path to a Flemming crop."""
    return "/".join(path.split("/")[-2:])


def select(a):
    adapt = pq.read_table(a.adapt, columns=["image_path", "speciesKey"]).to_pydict()
    opts = pv.ConvertOptions(include_columns=["filename", "level", "prediction", "confidence"], column_types={"prediction": pa.string()})
    rows = pv.read_csv(a.predictions, convert_options=opts)
    rows = rows.filter(pc.equal(rows["level"], 0)).to_pydict()
    pred = {crop(f): (p, c) for f, p, c in zip(rows["filename"], rows["prediction"], rows["confidence"], strict=True)}
    paths = adapt["image_path"]
    if missing := [p for p in paths if p not in pred]:
        raise SystemExit(f"{len(missing)} adapt crops have no prediction, e.g. {missing[0]}")
    label = np.array([pred[p][0] for p in paths])
    truth = np.array([str(s) for s in adapt["speciesKey"]])
    conf = np.array([pred[p][1] for p in paths], dtype=np.float32)  # dev/065's float32 softmax
    cut = float(np.quantile(conf, 1.0 - a.keep_frac))
    idx = np.flatnonzero(conf >= cut)
    keys = (truth if a.true_labels else label)[idx]
    out = pa.table(
        {
            "image_path": [paths[i] for i in idx],
            "speciesKey": keys.tolist(),
            "filename": [paths[i].rsplit("/", 1)[-1] for i in idx],
            "set": ["2"] * len(idx),
        }
    )
    pq.write_table(out, a.out)
    summary = {
        "labels": "true" if a.true_labels else "pseudo",
        "n_adapt": len(paths),
        "keep_frac": a.keep_frac,
        "confidence_cut": cut,
        "n_kept": len(idx),
        "n_distinct_species": len(set(keys)),
        "confidence_percentiles": {str(q): float(np.quantile(conf, q)) for q in (0.1, 0.5, 0.9, 0.99)},
        "pseudo_accuracy_all": float((label == truth).mean()),  # diagnostics only
        "pseudo_accuracy_kept": float((label[idx] == truth[idx]).mean()),
    }
    Path(f"{a.out}.summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def merge(a):
    cols = ["filename", "set", *KEYS]
    real = pq.read_table(a.global_parquet, columns=cols).select(cols).replace_schema_metadata(None)
    labels = pq.read_table(a.labels, columns=["image_path", "speciesKey"]).to_pydict()
    paths, species = [crop(p) for p in labels["image_path"]], [str(s) for s in labels["speciesKey"]]
    n_labels = len(paths)
    if a.fold:
        traps = [m.group(1) if (m := TRAP_RE.search(p)) else None for p in paths]
        if set(traps) != set(TRAPS):
            raise SystemExit(f"unexpected trap names: {sorted(set(traps), key=str)}")
        keep = [t not in FOLDS[a.fold] for t in traps]
        paths, species = list(compress(paths, keep)), list(compress(species, keep))
    n_fold = len(paths)

    # The floor counts real rows only, so label rows cannot add a species to the vocabulary.
    counts = pc.value_counts(real["speciesKey"].filter(pc.fill_null(pc.not_equal(real["set"], "0"), True)))
    vocab = counts.field("values").filter(pc.greater_equal(counts.field("counts"), a.min_count))
    keep = pc.is_in(pa.array(species, pa.string()), value_set=vocab).to_pylist()
    paths, species = list(compress(paths, keep)), list(compress(species, keep))
    if not paths:
        raise SystemExit("no label rows in the vocabulary")
    names = [p.rsplit("/", 1)[-1] for p in paths]

    mine = real.filter(pc.is_in(real["speciesKey"], value_set=pa.array(sorted(set(species)))))
    tax = mine.group_by("speciesKey", use_threads=False).aggregate([(k, "first") for k in KEYS[1:]])  # first real row
    tax = {r["speciesKey"]: r for r in tax.to_pylist()}
    dest = [f"{s}/{n}" for s, n in zip(species, names, strict=True)]
    taken = pc.binary_join_element_wise(mine["speciesKey"], mine["filename"], "/").combine_chunks()
    if clash := pc.sum(pc.is_in(pa.array(dest), value_set=taken)).as_py():
        raise SystemExit(f"{clash} crops would overwrite global images")
    stage = {}
    for d, p in zip(dest, paths, strict=True):
        stage.setdefault(d, p)  # the same crop filed under two species folders: identical files

    n = len(paths)
    block = pa.table(
        {"filename": names, "set": ["2"] * n, "speciesKey": species, **{k: [tax[s][f"{k}_first"] for s in species] for k in KEYS[1:]}}
    ).cast(real.schema)
    reps = max(1, round(a.target_frac * real.num_rows / ((1 - a.target_frac) * n))) if a.target_frac > 0 else 1
    combined = pa.concat_tables([real, *[block] * reps])
    train = pc.and_(pc.is_in(real["speciesKey"], value_set=vocab), pc.invert(pc.is_in(real["set"], value_set=pa.array(["0", "1"]))))
    summary = {
        "n_real": real.num_rows,
        "n_pseudo_unique": n,
        "replication": reps,
        "n_pseudo_rows": n * reps,
        "n_combined": combined.num_rows,
        "pseudo_frac": round(n * reps / combined.num_rows, 4),
        "species_real": pc.count_distinct(real["speciesKey"]).as_py(),
        "species_combined": pc.count_distinct(combined["speciesKey"]).as_py(),
        "min_img_per_spc_applied_to_real_only": a.min_count,
        "balanced": False,
        "fold": a.fold,
        "n_labels_in": n_labels,
        "n_after_fold": n_fold,
        "train_share": round(n * reps / (pc.sum(train).as_py() + n * reps), 4),
        "n_staged": len(stage),
    }
    print(json.dumps(summary, indent=2))
    if summary["species_combined"] > summary["species_real"]:
        raise SystemExit("label rows introduced new species")
    pq.write_table(combined, a.out)
    Path(f"{a.out}.summary.json").write_text(json.dumps(summary, indent=2))
    Path(f"{a.out}.staging.tsv").write_text("".join(f"{p}\t{d}\n" for d, p in stage.items()))


def pool(a):
    folds = TRAPS if a.folds == "trap" else [f"site-{s}" for s in ("LV", "OH", "SS")]
    ids, seen, header = {}, set(), None
    with open(a.out, "w", newline="") as f:
        out = csv.writer(f)
        for fold in folds:
            with open(a.pattern.format(fold=fold), newline="") as g:
                rows = csv.reader(g)
                if header is None:
                    header = next(rows)
                    out.writerow(header)
                elif next(rows) != header:
                    raise SystemExit(f"{fold}: columns differ")
                i, j = header.index("instance_id"), header.index("filename")
                for r in rows:
                    seen.add(crop(r[j]))
                    if TRAP_RE.search(r[j]).group(1) in FOLDS[fold]:
                        r[i] = str(ids.setdefault(crop(r[j]), len(ids)))
                        out.writerow(r)
    if len(ids) != len(seen):
        raise SystemExit(f"{len(seen) - len(ids)} crops have no held-out prediction")
    print(f"{len(ids):,} crops from {len(folds)} folds -> {a.out}")


def truth(a):
    """Relabel a label parquet's crops with their trap labels (the same crops, true species)."""
    trap = pq.read_table(a.trap, columns=["image_path", "speciesKey"]).to_pydict()
    true = {crop(p): str(s) for p, s in zip(trap["image_path"], trap["speciesKey"], strict=True)}
    paths = [crop(p) for p in pq.read_table(a.labels, columns=["image_path"]).column("image_path").to_pylist()]
    if missing := [p for p in paths if p not in true]:
        raise SystemExit(f"{len(missing)} crops are not in the trap parquet, e.g. {missing[0]}")
    keys = [true[p] for p in paths]
    out = {"image_path": paths, "speciesKey": keys, "filename": [p.rsplit("/", 1)[-1] for p in paths], "set": ["2"] * len(paths)}
    pq.write_table(pa.table(out), a.out)
    print(f"{len(paths):,} crops relabelled, {len(set(keys))} true species -> {a.out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(required=True)
    p = sub.add_parser("select", help="dev/065: gate the teacher's predictions on the adapt crops.")
    p.add_argument("--predictions", type=Path, required=True, help="mini_metric.csv (or predictions.csv) of the trap crops.")
    p.add_argument("--adapt", type=Path, required=True, help="lepinet's flemming_adapt.parquet.")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--keep-frac", type=float, default=0.30)
    p.add_argument("--true-labels", action="store_true", help="Label the kept crops with their trap labels (T1rep).")
    p.set_defaults(run=select)
    p = sub.add_parser("merge", help="dev/066 (and dev/089 with --fold): mini_trainer training parquet.")
    p.add_argument("--global-parquet", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True, help="select's output, or a lepinet pseudo/label parquet.")
    p.add_argument("--out", type=Path, required=True, help="Parquet next to the images/ folder it reads.")
    p.add_argument("--target-frac", type=float, default=0.02, help="0 for one copy of each label row.")
    p.add_argument("--fold", choices=list(FOLDS), help="Drop this trap's (or site's) label rows first.")
    p.add_argument("--min-count", type=int, default=50)
    p.set_defaults(run=merge)
    p = sub.add_parser("truth", help="The same crops with their trap labels (e.g. lepinet's gated crops, true labels).")
    p.add_argument("--labels", type=Path, required=True, help="A label parquet with image_path (e.g. pseudo_strict.parquet).")
    p.add_argument("--trap", type=Path, required=True, help="lepinet's flemming_names_labels.parquet.")
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(run=truth)
    p = sub.add_parser("pool", help="dev/089 pool: one held-out prediction per crop.")
    p.add_argument("--pattern", required=True, help="Path to each fold's predictions with a {fold} placeholder.")
    p.add_argument("--folds", choices=["trap", "site"], required=True)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(run=pool)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
