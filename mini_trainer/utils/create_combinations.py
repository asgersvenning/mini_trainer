import csv
import json
import os
import re
from argparse import ArgumentParser
from collections import OrderedDict
from itertools import chain
from pathlib import Path
from typing import Any

import torch
from torch import nn

from mini_trainer.integrations.gbif import TAXONOMY_KEYS, create_taxonomy, labels_from_taxonomy
from mini_trainer.integrations.parquet import KCOLUMNS, get_keys, iter_parquet
from mini_trainer.modeling import Classifier, classification_module

SCIENTIFICNAME_OR_NUMBER = re.compile(r"^(\d+|[\dA-Z]+|\w+(\s+\w+)?)$")


def detect_sep(content: str, options: tuple[str, ...] = (",", ";", r"\t")):
    counts = {o: content.count(o) for o in options}
    max_count = max(counts.values())
    if max_count == 0:
        return None
    candidates = [k for k, v in counts.items() if v >= max_count]
    match len(candidates):
        case 0:
            return None
        case 1:
            return candidates[0]
        case _:
            raise RuntimeError("Unable to automatically determine line separator in:\n" + content)


def is_taxalist(content: str | list[str]):
    if isinstance(content, str):
        content = [content]
    return all(map(lambda s: bool(re.match(SCIENTIFICNAME_OR_NUMBER, s.strip())), content))


def labels_from_json(file: str | dict, levels: int | list[int] | None = None, interactive: bool = True):
    if isinstance(file, (str, Path)):
        with open(file) as f:
            data = json.load(f)
    else:
        data = file

    for key in ("labels", "label"):
        if key in data:
            raw_labels = data[key]
            if isinstance(raw_labels, dict):
                raw_labels = list(raw_labels.values())
            if raw_labels and isinstance(raw_labels[0], (list, tuple)):
                assert all(isinstance(e, (list, tuple)) for e in raw_labels)
                assert all(all(isinstance(ei, (str, int)) for ei in e) for e in raw_labels)
                assert len(set(map(len, raw_labels))) == 1
                labels = list(set(map(lambda x: tuple(map(str, x)), raw_labels)))
                detected_levels = list(range(len(labels[0])))
                return labels, detected_levels
            classes = sorted(set(map(str, raw_labels)))
            return labels_from_taxalist(classes, levels=levels, interactive=interactive)

    if "cls2idx" in data:
        cls2idx: dict[str, int | dict[str, int]] = data["cls2idx"]
        if "0" in cls2idx and isinstance((ld := cls2idx["0"]), dict):
            detected_levels = len(cls2idx) if levels is None else levels
            classes = list(ld.keys())
        else:
            detected_levels = levels
            classes = list(cls2idx.keys())
        return labels_from_taxalist(classes, levels=detected_levels, interactive=interactive)

    raise KeyError(f'File/Object {file} should contain "labels", "label", or "cls2idx" field if a JSON')


def labels_from_model(file: str | OrderedDict[str, torch.Tensor | Any] | nn.Module, interactive: bool = True):
    if not isinstance(file, nn.Module):
        model, _ = Classifier.build(weights=file)
    else:
        model = file
    head = classification_module(model)
    metadata = head.metadata
    return labels_from_json(metadata, interactive=interactive)


def labels_from_csv(file: str, interactive: bool = True):
    with open(file) as f:
        reader = csv.reader(f)
        headers = next(reader)
        data = {k: [] for k in headers}
        for row in reader:
            for c, v in zip(headers, row):
                data[c].append(v)
    if "level" in data:
        data = {k: [vi for lvl, vi in zip(data["level"], v) if lvl in (0, "0", "species", "Species")] for k, v in data.items()}
    if "label" not in data and "prediction" not in data:
        raise RuntimeError(
            f"File {file} contains unknown CSV schema:\n"
            + " | ".join(f"{k} : {type(v[0] if v else None)}" for k, v in data.items())
            + "\nExpected a 'label' column of type str/int"
        )
    cols = ["label", "prediction"]
    return labels_from_taxalist(list(set(map(str, chain.from_iterable(data.get(c, []) for c in cols)))), interactive=interactive)


def labels_from_txt(file: str, interactive: bool = True):
    with open(file) as f:
        content = [f for f in map(str.strip, f.readlines()) if f]
    if len(content) == 0:
        raise RuntimeError(f"Empty file {file}")
    if len(content) == 1:
        content = content[0].split(detect_sep(content[0]))
    return labels_from_taxalist(content, interactive=interactive)


def labels_from_parquet(file: str, levels: int | list[int] | str | None = None, interactive: bool = True):
    if levels is None:
        num_levels = 3
    elif isinstance(levels, (list, tuple)):
        num_levels = max(TAXONOMY_KEYS.index(lvl) + 1 if isinstance(lvl, str) else lvl for lvl in levels)
    elif isinstance(levels, str):
        num_levels = TAXONOMY_KEYS.index(levels) + 1
    else:
        num_levels = levels

    rows = [tuple(get_keys(row)[:num_levels]) for row in iter_parquet(file, KCOLUMNS)]
    labels = sorted(set(rows), key=lambda x: x[::-1])
    return labels, list(range(num_levels))


def labels_from_dir(dir_path: str, levels: int | list[int] | None = None, interactive: bool = True):
    subdirs = sorted(d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d)) and not d.startswith("."))
    if not subdirs:
        raise RuntimeError(f"Directory {dir_path} does not contain any class subdirectories.")
    return labels_from_taxalist(subdirs, levels=levels, interactive=interactive)


def labels_from_taxalist(
    taxa: list[str], levels: list[int] | list[str] | list[int | str] | str | int | None = None, interactive: bool = True
) -> tuple[list[tuple[str, ...]], list[int]]:
    if isinstance(levels, (str, int)):
        levels = [levels]
    if isinstance(levels, list):
        levels = max(TAXONOMY_KEYS.index(lvl) + 1 if isinstance(lvl, str) else lvl for lvl in levels)
    if is_taxalist(taxa):
        txl = ""
        if interactive:
            txl = input(f"Which levels do you want to include (default={TAXONOMY_KEYS[levels - 1] if levels is not None else '??'}):")
        txl = txl or (levels - 1 if isinstance(levels, int) else levels)
        labels = list(labels_from_taxonomy(create_taxonomy(taxa, levels=txl)).values())
    else:
        raise RuntimeError(f"Unknown content: {taxa[: min(len(taxa), 10)]}")
    return labels, list(range(len(labels[0])))


def labels_from_source(
    source: str | Path | dict, levels: int | list[int] | None = None, interactive: bool = False
) -> tuple[list[tuple[str, ...]], list[int]]:
    if isinstance(source, dict):
        if "data_index" in source and isinstance(source["data_index"], (str, Path)) and os.path.exists(str(source["data_index"])):
            return labels_from_json(str(source["data_index"]), levels=levels, interactive=interactive)
        if "path" in source and source["path"]:
            return labels_from_source(source["path"], levels=levels, interactive=interactive)
        if any(k in source for k in ("labels", "label", "cls2idx")):
            return labels_from_json(source, levels=levels, interactive=interactive)
        raise ValueError(f"Cannot resolve dataset source from dict: {source}")

    src_path = Path(source).expanduser().resolve()
    if not src_path.exists():
        raise FileNotFoundError(f"Dataset source '{src_path}' does not exist.")

    if src_path.is_dir():
        if (src_path / "data_index.json").exists():
            return labels_from_json(str(src_path / "data_index.json"), levels=levels, interactive=interactive)
        if (src_path / "class_spec.json").exists():
            return labels_from_json(str(src_path / "class_spec.json"), levels=levels, interactive=interactive)
        return labels_from_dir(str(src_path), levels=levels, interactive=interactive)

    ext = src_path.suffix.lower()
    match ext:
        case ".pt" | ".pth":
            return labels_from_model(str(src_path), interactive=interactive)
        case ".json":
            return labels_from_json(str(src_path), levels=levels, interactive=interactive)
        case ".parquet":
            return labels_from_parquet(str(src_path), levels=levels, interactive=interactive)
        case ".csv":
            return labels_from_csv(str(src_path), interactive=interactive)
        case _:
            return labels_from_txt(str(src_path), interactive=interactive)


def create_combinations_file(
    sources: list[str | Path | dict] | str | Path | dict,
    output: str | Path | None = None,
    levels: int | list[int] | None = None,
    interactive: bool = False,
) -> tuple[list[tuple[str, ...]], list[int]]:
    if not isinstance(sources, list):
        sources = [sources]

    all_labels: set[tuple[str, ...]] = set()
    detected_levels: list[int] | None = None

    for src in sources:
        labels, lvls = labels_from_source(src, levels=levels, interactive=interactive)
        all_labels.update(labels)
        if detected_levels is None or len(lvls) > len(detected_levels):
            detected_levels = lvls

    if not all_labels:
        raise ValueError(f"No labels could be extracted from sources: {sources}")

    if detected_levels is None:
        detected_levels = list(range(len(next(iter(all_labels)))))

    sorted_labels = sorted(all_labels, key=lambda x: x[::-1])

    if output is not None:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        colnames = [TAXONOMY_KEYS[i] for i in detected_levels]
        with open(output_path, "w", encoding="utf8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(colnames)
            for label in sorted_labels:
                writer.writerow(label)

    return sorted_labels, detected_levels


def main(file: str, output: str | None = None, interactive: bool = True):
    return create_combinations_file(sources=file, output=output, interactive=interactive)[0]


def cli():
    parser = ArgumentParser(
        prog="class2combinations",
        description=(
            "Create a hierarchical combinations file from a list of species, a model or a class specification created by mini_trainer."
        ),
    )
    parser.add_argument(
        "-i",
        "--input",
        dest="file",
        type=str,
        required=True,
        help="Source for creating the combinations, "
        "either a file with a list of species, a model,"
        "or a class specification JSON created by mini_trainer.",
    )
    parser.add_argument("-o", "--output", type=str, required=True, help="Output file.")
    parser.add_argument("-y", "-Y", action="store_false", dest="interactive", help="Automatically accept the recommended taxa levels.")

    return vars(parser.parse_args())


def run():
    main(**cli())


if __name__ == "__main__":
    run()
