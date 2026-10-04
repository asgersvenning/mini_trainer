"""Frozen, train-selected species cohorts; no image decoding or taxonomy requests."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def digest(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def table_digest(frame):
    """Hash ordered column names and values, independent of Parquet metadata."""
    sha = hashlib.sha256()
    sha.update((json.dumps(list(frame.columns), ensure_ascii=True, separators=(",", ":")) + "\n").encode())
    for row in frame.itertuples(index=False, name=None):
        sha.update((json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n").encode())
    return sha.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def species_table(frame):
    required = ["speciesKey", "familyKey", "genusKey", "set"]
    if frame[required].isna().any().any():
        raise ValueError("Missing taxonomy or source split")
    if not frame["set"].astype(str).isin([str(i) for i in range(10)]).all():
        raise ValueError("Expected original numeric source splits 0 through 9")
    train = frame[~frame["set"].astype(str).isin(["0", "1"])]
    if (train.groupby("speciesKey")[["familyKey", "genusKey"]].nunique() > 1).any().any():
        raise ValueError("Conflicting training taxonomy")
    table = train.groupby("speciesKey", sort=True).agg(
        train=("set", "size"), familyKey=("familyKey", "first"), genusKey=("genusKey", "first")
    )
    # Stable tie-breaking is species-key order, independent of Parquet row order.
    table["quartile"] = pd.qcut(table.train.rank(method="first"), 4, labels=False)
    return table


def select_species(table, count, seed):
    """Nested proportional family x abundance allocation by largest deficit."""
    if not 4 <= count <= len(table):
        raise ValueError("Species count must be between four and available training species")
    rng = np.random.default_rng(seed)
    groups = [list(rng.permutation(group.index.to_numpy())) for _, group in table.groupby(["familyKey", "quartile"], sort=True)]
    sizes = np.array([len(group) for group in groups])
    used = np.zeros(len(groups), dtype=int)
    chosen = []
    for step in range(count):
        deficit = (step + 1) * sizes / sizes.sum() - used
        deficit[used == sizes] = -np.inf
        index = int(deficit.argmax())
        chosen.append(str(groups[index][used[index]]))
        used[index] += 1
    return chosen


def describe(table):
    values = np.sort(table.train.to_numpy())
    n = len(values)
    return {
        "species": n,
        "families": int(table.familyKey.nunique()),
        "genera": int(table.genusKey.nunique()),
        "train_images": int(values.sum()),
        "train_quantiles": {str(q): float(np.quantile(values, q)) for q in [0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1]},
        "gini": float(2 * (np.arange(1, n + 1) * values).sum() / (n * values.sum()) - (n + 1) / n),
        "family_species_counts": {str(k): int(v) for k, v in table.familyKey.value_counts().items()},
    }


def select_families(table, image_budget, seed):
    """Seeded complete-family packing; never truncate a family or rebalance it."""
    counts = table.groupby("familyKey", sort=True).train.sum()
    chosen, used = [], 0
    for family in np.random.default_rng(seed).permutation(counts.index):
        if used + counts[family] <= image_budget:
            chosen.append(str(family))
            used += int(counts[family])
    return chosen


def hierarchy_description(table):
    """Describe natural branching and abundance without using held-out images."""
    if (table.groupby("genusKey").familyKey.nunique() > 1).any():
        raise ValueError("Genus belongs to multiple families")
    genera = table.groupby("genusKey").size()
    families = table.groupby("familyKey").genusKey.nunique()
    return {
        **describe(table),
        "singleton_genera": int((genera == 1).sum()),
        "species_in_singleton_genera_fraction": float((genera == 1).sum() / len(table)),
        "species_per_genus": {str(k): int(v) for k, v in genera.items()},
        "genera_per_family": {str(k): int(v) for k, v in families.items()},
    }


def hierarchy_spec(table, classes):
    """Freeze leaf-first rank vocabularies and contiguous child-to-parent maps."""
    rows = table.loc[classes]
    keys = [classes, sorted(rows.genusKey.unique()), sorted(rows.familyKey.unique())]
    mappings = {str(i): {key: j for j, key in enumerate(values)} for i, values in enumerate(keys)}
    masks = [
        [mappings["1"][key] for key in rows.genusKey],
        [mappings["2"][key] for key in rows.groupby("genusKey").familyKey.first().reindex(keys[1])],
    ]
    counts = [rows.train.astype(int).tolist()]
    for column, values in zip(["genusKey", "familyKey"], keys[1:]):
        counts.append(rows.groupby(column).train.sum().reindex(values).astype(int).tolist())
    return {"cls2idx": mappings, "num_classes": list(map(len, keys)), "masks": masks, "counts": counts}


def prepare_data(config, root):
    source = config["parquet"]
    columns = ["speciesKey", "familyKey", "genusKey", "set"]
    frame = pd.read_parquet(source, columns=columns).astype(str)
    table = species_table(frame)
    candidates = {}
    for n in [256, 512, 1024]:
        if n <= len(table):
            candidates[str(n)] = describe(table.loc[select_species(table, n, config["selection_seed"])])
    if config.get("hierarchy"):
        families = list(map(str, config["cohort_families"]))
        if families != select_families(table, config["train_image_budget"], config["selection_seed"]):
            raise ValueError("Selected families differ from frozen selection rule")
        if len(set(families)) != len(families) or set(families) - set(table.familyKey):
            raise ValueError("Unknown or duplicate cohort family")
        selected = table.index[table.familyKey.isin(families)].tolist()
        if len(selected) != config["species"]:
            raise ValueError("Complete-family species count differs from frozen protocol")
    else:
        selected = select_species(table, config["species"], config["selection_seed"])
    classes = sorted(selected)
    mapping = {key: i for i, key in enumerate(classes)}
    # Predicate pushdown avoids loading all image identifiers into memory.
    samples = pd.read_parquet(source, columns=columns + ["gbifID", "filename", "scientificName"], filters=[("speciesKey", "in", selected)])
    if samples.isna().any().any():
        raise ValueError("Missing sample identity or taxonomy")
    samples = samples.astype(str)
    if samples.duplicated(["speciesKey", "filename"]).any():
        raise ValueError("Duplicate image paths in selected cohort")
    samples["split"] = samples["set"].map(lambda value: "test" if value == "0" else "validation" if value == "1" else "train")
    if (samples.groupby("gbifID")["split"].nunique() > 1).any():
        raise ValueError("Observation crosses source partitions; preserve evidence and resolve upstream")
    if (samples.groupby("speciesKey")[["familyKey", "genusKey"]].nunique() > 1).any().any():
        raise ValueError("Conflicting selected taxonomy")
    for row in samples[["speciesKey", "filename"]].itertuples(index=False):
        if Path(row.filename).name != row.filename or Path(row.speciesKey).name != row.speciesKey:
            raise ValueError("Expected simple species directory and image filename")
    samples["label"] = samples.speciesKey.map(mapping)
    samples = samples.sort_values(["split", "speciesKey", "filename"], kind="stable").reset_index(drop=True)
    samples["sample_id"] = samples.speciesKey + "/" + samples.filename
    samples.to_parquet(root / "samples.parquet", index=False)
    counts = table.loc[classes, "train"].astype(int).tolist()
    spec = {"num_classes": len(classes), "cls2idx": mapping, "counts": counts, "resize_size": config["size"]}
    if config.get("hierarchy"):
        spec["hierarchy"] = hierarchy_spec(table, classes)
    write_json(root / "classes.json", spec)
    support = samples.groupby(["speciesKey", "split"]).size().unstack(fill_value=0).reindex(classes, fill_value=0)
    support = table.loc[classes].join(support.rename(columns={"train": "train_support"}))
    support.to_csv(root / "species.csv")
    write_json(
        root / "selection.json",
        {
            "source_sha256": digest(source),
            "selection_seed": config["selection_seed"],
            "full": describe(table),
            "candidates": candidates,
            "selected": describe(table.loc[classes]),
            **(
                {
                    "hierarchy_full": hierarchy_description(table),
                    "hierarchy_selected": hierarchy_description(table.loc[classes]),
                    "samples_content_sha256": table_digest(samples),
                }
                if config.get("hierarchy")
                else {}
            ),
            "split_counts": samples.split.value_counts().to_dict(),
            "cross_partition_observations": 0,
            "split_species_support": samples.groupby("split").speciesKey.nunique().to_dict(),
        },
    )
