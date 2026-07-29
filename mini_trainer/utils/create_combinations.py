import csv
import json
import os
import re
from argparse import ArgumentParser
from collections import OrderedDict
from itertools import chain
from typing import Any, cast

import torch
from torch import nn

from mini_trainer.integrations.gbif import TAXONOMY_KEYS, create_taxonomy, labels_from_taxonomy
from mini_trainer.modeling import Classifier, classification_module

SCIENTIFICNAME_OR_NUMBER = re.compile(r"^(\d+|\w+(\s+\w+)?)$")


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
    return all(map(lambda s: bool(re.match(SCIENTIFICNAME_OR_NUMBER, s)), content))


def labels_from_json(file: str | dict, interactive: bool = True):
    if isinstance(file, str):
        with open(file) as f:
            data = json.load(f)
    else:
        data = file
    if "label" in data:
        labels = list(data["label"].values())
        # Check that labels is a list of tuples/lists of the same lengths containing strings
        assert all(map(lambda e: isinstance(e, (list, tuple)), labels))
        assert all(map(lambda e: all(map(lambda ei: isinstance(ei, str), e)), labels))
        assert len(set(map(len, labels))) == 1
        # Cast to asserted type
        labels = cast(list[list | tuple[str, ...]], labels)
        labels = list(map(tuple, labels))
        levels = list(range(len(labels[0])))
    else:
        if "cls2idx" not in data:
            raise KeyError(f'File {file} should contain a "label" or "cls2idx" field if a JSON')
        cls2idx: dict[str, int | dict[str, int]] = data["cls2idx"]
        if "0" in cls2idx and isinstance((ld := cls2idx["0"]), dict):
            levels = len(cls2idx)
            classes = list(ld.keys())
        else:
            levels = None
            classes = list(cls2idx.keys())
        labels, levels = labels_from_taxalist(classes, levels=levels, interactive=interactive)
    return labels, levels


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


def main(file: str, output: str | None = None, interactive: bool = True):
    if not os.path.exists(file):
        raise FileNotFoundError(f"Supplied source file {file} does not exist.")
    if not os.path.isfile(file):
        raise ValueError(f"Supplied source file {file} is not a file.")

    _, ext = os.path.splitext(file)
    match ext.lower():
        case ".pt" | ".pth":
            label_retriever = labels_from_model
        case ".json":
            label_retriever = labels_from_json
        case ".csv":
            label_retriever = labels_from_csv
        case _:
            label_retriever = labels_from_txt

    labels, levels = label_retriever(file, interactive=interactive)
    labels = sorted(labels, key=lambda x: x[::-1])

    if output is not None:
        colnames = [TAXONOMY_KEYS[i] for i in levels]
        with open(output, "w", encoding="utf8") as f:
            writer = csv.writer(f)
            writer.writerow(colnames)
            for label in labels:
                writer.writerow(label)

    return labels


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
