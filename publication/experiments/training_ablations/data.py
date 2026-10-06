"""Frozen, train-selected species cohorts; no image decoding or taxonomy requests."""

import hashlib
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

RANKS = ("species", "genus", "family", "order", "class")  # Leaf-first, as GBIF <rank>Key columns


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


def informative_ranks(taxonomy):
    """Rank key columns chosen by mini_trainer's rule: ranks with more than one taxon in the vocabulary."""
    from mini_trainer.integrations.gbif import TAXONOMY_KEYS, select_levels

    columns = [f"{rank}Key" for rank in TAXONOMY_KEYS if f"{rank}Key" in taxonomy]
    rows = [
        OrderedDict((c.removesuffix("Key"), (str(v), "")) for c, v in zip(columns, row))
        for row in taxonomy[columns].itertuples(index=False)
    ]
    return [f"{level}Key" for level in select_levels(None, rows)]


def check_ranks(table, ranks):
    """A cohort's configured ranks must be exactly the informative ranks of its classes."""
    found = [key.removesuffix("Key") for key in informative_ranks(table.reset_index())]
    if found != list(ranks):
        raise ValueError(f"Configured ranks {list(ranks)} differ from the cohort's informative ranks {found}")


def species_table(frame, parents=("genusKey", "familyKey")):
    """Training counts and parent rank keys per species."""
    parents = list(parents)
    required = ["speciesKey", *parents, "set"]
    if frame[required].isna().any().any():
        raise ValueError("Missing taxonomy or source split")
    if not frame["set"].astype(str).isin([str(i) for i in range(10)]).all():
        raise ValueError("Expected original numeric source splits 0 through 9")
    train = frame[~frame["set"].astype(str).isin(["0", "1"])]
    if (train.groupby("speciesKey")[parents].nunique() > 1).any().any():
        raise ValueError("Conflicting training taxonomy")
    table = train.groupby("speciesKey", sort=True).agg(train=("set", "size"), **{key: (key, "first") for key in parents})
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


def hierarchy_spec(table, classes, ranks=RANKS[:3]):
    """Freeze leaf-first rank vocabularies, contiguous child-to-parent maps and training counts."""
    rows = table.loc[classes].reset_index(drop=True).assign(speciesKey=classes)
    columns = [f"{rank}Key" for rank in ranks]
    keys = [classes, *(sorted(rows[column].unique()) for column in columns[1:])]
    mappings = {str(i): {key: j for j, key in enumerate(values)} for i, values in enumerate(keys)}
    masks = [
        [mappings[str(i + 1)][key] for key in rows.groupby(child)[parent].first().reindex(keys[i])]
        for i, (child, parent) in enumerate(zip(columns, columns[1:]))
    ]
    counts = [rows.groupby(column).train.sum().reindex(values).astype(int).tolist() for column, values in zip(columns, keys)]
    return {"ranks": list(ranks), "cls2idx": mappings, "num_classes": list(map(len, keys)), "masks": masks, "counts": counts}


def lower_unique_support(samples, table, cap, seed):
    """Cap the least-supported training third and restore original class draws."""
    if cap is None:
        return samples, None
    if not isinstance(cap, int) or cap < 1:
        raise ValueError("train_support_cap must be a positive integer or null")
    counts = table.train
    ordered = sorted(counts.index, key=lambda species: (int(counts[species]), str(species)))
    tail = ordered[: max(1, (len(ordered) + 2) // 3)]
    if any(int(counts[species]) < cap for species in tail):
        raise ValueError("train_support_cap exceeds support in the lowest-frequency third")

    rng = np.random.default_rng(seed)
    train = samples[samples.split == "train"]
    held_out = samples[samples.split != "train"]
    draws = []
    for species, group in train.groupby("speciesKey", sort=True):
        original_count = len(group)
        if species not in tail or original_count <= cap:
            draws.append(group)
            continue
        selected = group.iloc[np.sort(rng.choice(original_count, size=cap, replace=False))]
        repeated = rng.choice(cap, size=original_count - cap, replace=True)
        indices = rng.permutation(np.concatenate([np.arange(cap), repeated]))
        draws.append(selected.iloc[indices])
    sampled_train = pd.concat(draws, ignore_index=True)
    result = pd.concat([sampled_train, held_out], ignore_index=True)
    result = result.sort_values(["split", "speciesKey", "sample_id"], kind="stable").reset_index(drop=True)
    unique = sampled_train.groupby("speciesKey").sample_id.nunique()
    if len(sampled_train) != len(train) or any(unique[species] != min(int(counts[species]), cap) for species in tail):
        raise RuntimeError("Support reduction violated its draw-count or support target")
    return result, {
        "method": "lowest_frequency_third_capped_with_replacement_to_original_class_draws",
        "cap": cap,
        "seed": seed,
        "tail_species": [str(species) for species in tail],
        "tail_species_count": len(tail),
        "original_train_draws": len(train),
        "unique_train_images_before": int(train.sample_id.nunique()),
        "unique_train_images_after": int(sampled_train.sample_id.nunique()),
        "train_draws_after": len(sampled_train),
        "held_out_rows_unchanged": len(held_out),
    }


def data_source(config):
    """The dataset a study config trains on and its source file; the one rule preparation and export share."""
    if config.get("data_index"):
        return "plantnet", config["data_index"]
    return "global_lepi", config["parquet"]


def prepare_data(config, root):
    dataset, source = data_source(config)
    if dataset == "plantnet":
        return prepare_index(config, root)
    # Read every rank the source supplies, so the configured ranks can be checked against the rule.
    keys = [f"{rank}Key" for rank in RANKS if f"{rank}Key" in pq.read_schema(source).names]
    columns = [*keys, "set"]
    frame = pd.read_parquet(source, columns=columns).astype(str)
    table = species_table(frame, keys[1:])
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
    if (samples.groupby("speciesKey")[keys[1:]].nunique() > 1).any().any():
        raise ValueError("Conflicting selected taxonomy")
    for row in samples[["speciesKey", "filename"]].itertuples(index=False):
        if Path(row.filename).name != row.filename or Path(row.speciesKey).name != row.speciesKey:
            raise ValueError("Expected simple species directory and image filename")
    samples["label"] = samples.speciesKey.map(mapping)
    samples = samples.sort_values(["split", "speciesKey", "filename"], kind="stable").reset_index(drop=True)
    samples["sample_id"] = samples.speciesKey + "/" + samples.filename
    samples, support_reduction = lower_unique_support(
        samples, table.loc[classes], config.get("train_support_cap"), config.get("train_support_seed", 0)
    )
    samples.to_parquet(root / "samples.parquet", index=False)
    counts = table.loc[classes, "train"].astype(int).tolist()
    spec = {"num_classes": len(classes), "cls2idx": mapping, "counts": counts, "resize_size": config["size"]}
    check_ranks(table.loc[classes], config["ranks"])
    if config.get("hierarchy"):
        spec["hierarchy"] = hierarchy_spec(table, classes, config["ranks"])
    write_json(root / "classes.json", spec)
    support = samples.groupby(["speciesKey", "split"]).size().unstack(fill_value=0).reindex(classes, fill_value=0)
    support = table.loc[classes].join(support.rename(columns={"train": "train_support"}))
    if support_reduction:
        support["unique_train_support"] = support.train
        support.loc[support_reduction["tail_species"], "unique_train_support"] = config["train_support_cap"]
        support["training_draws"] = support.train_support
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
            **({"support_reduction": support_reduction} if support_reduction else {}),
        },
    )


def prepare_index(config, root):
    """Reuse an already corrected hierarchical index without taxonomy lookups."""
    source = Path(config["data_index"])
    index = json.loads(source.read_text())
    fields = [index[key] for key in ("path", "split", "label")]
    if not fields[0] or len({len(values) for values in fields}) != 1:
        raise ValueError("Index fields must have equal nonzero lengths")
    rows = []
    for path, split, labels in zip(*fields):
        relative = Path(path)
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ValueError("Index image paths must be relative without parent traversal")
        if (
            split not in ("train", "validation", "test")
            or not 3 <= len(labels) <= len(RANKS)
            or any(v is None or str(v) == "" for v in labels)
        ):
            raise ValueError("Expected supplied splits and leaf-first labels from species to at least family")
        rows.append((relative.as_posix(), split, *map(str, labels)))
    if len({len(row) for row in rows}) != 1:
        raise ValueError("Index labels must cover the same ranks for every image")
    keys = [f"{rank}Key" for rank in RANKS[: len(rows[0]) - 2]]
    samples = pd.DataFrame(rows, columns=["sample_id", "split", *keys])
    if samples.sample_id.duplicated().any() or samples.sample_id.map(lambda p: Path(p).name).duplicated().any():
        raise ValueError("Duplicate image identity in corrected index")
    if (samples.groupby("speciesKey")[keys[1:]].nunique() > 1).any().any():
        raise ValueError("Conflicting taxonomy in corrected index")
    train = samples[samples.split == "train"]
    table = train.groupby("speciesKey", sort=True).agg(train=("split", "size"), **{key: (key, "first") for key in keys[1:]})
    check_ranks(table, config["ranks"])
    if set(samples.speciesKey) != set(table.index):
        raise ValueError("Held-out class has no training examples")
    description = hierarchy_description(table)
    classes = table.index.tolist()
    mapping = {key: i for i, key in enumerate(classes)}
    samples["label"] = samples.speciesKey.map(mapping)
    samples = samples.sort_values(["split", "speciesKey", "sample_id"], kind="stable").reset_index(drop=True)
    missing = [p for p in samples.sample_id if not (Path(config["images"]) / p).is_file()]
    if missing:
        raise ValueError(f"Corrected dataset has {len(missing)} missing images; first: {missing[0]}")
    source_audit = audit_source_metadata(samples, config, root)
    samples, support_reduction = lower_unique_support(samples, table, config.get("train_support_cap"), config.get("train_support_seed", 0))
    samples.to_parquet(root / "samples.parquet", index=False)
    write_json(
        root / "classes.json",
        {
            "num_classes": len(classes),
            "cls2idx": mapping,
            "counts": table.train.tolist(),
            "resize_size": config["size"],
            "hierarchy": hierarchy_spec(table, classes, config["ranks"]),
        },
    )
    support = samples.groupby(["speciesKey", "split"]).size().unstack(fill_value=0)
    support_table = table.join(support.rename(columns={"train": "train_support"}))
    if support_reduction:
        support_table["unique_train_support"] = support_table.train
        support_table.loc[support_reduction["tail_species"], "unique_train_support"] = config["train_support_cap"]
        support_table["training_draws"] = support_table.train_support
    support_table.to_csv(root / "species.csv")
    write_json(
        root / "selection.json",
        {
            "source_audit": source_audit,
            "source_sha256": digest(source),
            "source": str(source),
            "selection": "all corrected-index classes and images",
            "selected": description,
            "hierarchy_selected": description,
            "samples_content_sha256": table_digest(samples),
            "split_counts": samples.split.value_counts().to_dict(),
            "split_species_support": samples.groupby("split").speciesKey.nunique().to_dict(),
            "cross_partition_observations": source_audit.get("cross_partition_observations"),
            **({"support_reduction": support_reduction} if support_reduction else {}),
            "observation_audit": "Inherited overlap retained; see source_audit"
            if source_audit["available"]
            else "Index lacks observation IDs; audit original metadata separately",
        },
    )


def audit_source_metadata(samples, config, root):
    """Describe inherited observation overlap and corrected source class merges."""
    path = config.get("source_metadata")
    if not path:
        return {"available": False}
    source = pd.read_csv(path, dtype=str)
    required = ["PN_hash", "PN_observation_id", "species_id", "split"]
    if source[required].isna().any().any() or source.PN_hash.duplicated().any():
        raise ValueError("Source metadata has missing identities or duplicate image hashes")
    selected = samples.assign(PN_hash=samples.sample_id.map(lambda p: Path(p).stem)).merge(
        source[required], on="PN_hash", how="left", validate="one_to_one", suffixes=("", "_source")
    )
    if selected.species_id.isna().any() or not selected.split.equals(selected.split_source.replace({"val": "validation"})):
        raise ValueError("Corrected index does not preserve source image identities and splits")
    mapping = selected.groupby(["species_id", "speciesKey"]).size().rename("images").reset_index()
    if (mapping.groupby("species_id").speciesKey.nunique() != 1).any():
        raise ValueError("An original class maps to multiple corrected species")
    mapping.to_csv(root / "source-class-map.csv", index=False)
    return {
        "available": True,
        "source_sha256": digest(path),
        "retained_images": len(selected),
        "excluded_images": len(source) - len(selected),
        "excluded_source_classes": sorted(set(source.species_id) - set(selected.species_id)),
        "merged_corrected_classes": int((mapping.groupby("speciesKey").species_id.nunique() > 1).sum()),
        "cross_partition_observations": int((selected.groupby("PN_observation_id").split.nunique() > 1).sum()),
    }
