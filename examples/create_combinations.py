import csv
import json
import os
import re
from argparse import ArgumentParser
from itertools import chain
from typing import cast

from mini_trainer.utils.gbif import TAXONOMY_KEYS, create_taxonomy, labels_from_taxonomy

SCIENTIFICNAME_OR_NUMBER = re.compile(r"^(\d+|\w+(\s+\w+)?)$")


def detect_sep(content: str, options: tuple[str] = (",", ";", r"\t")):
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


def labels_from_json(file: str):
    with open(file) as f:
        data = json.load(f)
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
        if "0" in cls2idx and isinstance(cls2idx["0"], dict):
            levels = len(cls2idx)
            classes = list(cls2idx["0"].keys())
        else:
            levels = 1
            classes = list(cls2idx.keys())
        levels_name = TAXONOMY_KEYS[levels - 1]
        labels = labels_from_taxonomy(
            create_taxonomy(classes, levels=input(f"Which levels do you want to include (default={levels_name}):") or levels_name)
        )
        labels = list(labels.values())
    return labels, levels


def labels_from_csv(file: str):
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
    return labels_from_taxalist(list(set(map(str, chain.from_iterable(data.get(c, []) for c in cols)))))


def labels_from_txt(file: str):
    with open(file) as f:
        content = [f for f in map(str.strip, f.readlines()) if f]
    if len(content) == 0:
        raise RuntimeError(f"Empty file {file}")
    if len(content) == 1:
        content = content[0].split(detect_sep(content[0]))
    return labels_from_taxalist(content)


def labels_from_taxalist(taxa: list[str]):
    if is_taxalist(taxa):
        labels = labels_from_taxonomy(
            create_taxonomy(taxa, levels=input("Which levels do you want to include (default=family):") or "family")
        )
        labels = list(labels.values())
        levels = list(range(len(labels[0])))
    else:
        raise RuntimeError(f"Unknown content: {taxa[: min(len(taxa), 10)]}")
    return labels, levels


def main(file: str, output: str | None = None):
    if not os.path.exists(file):
        raise FileNotFoundError(f"Supplied source file {file} does not exist.")
    if not os.path.isfile(file):
        raise ValueError(f"Supplied source file {file} is not a file.")
    _, ext = os.path.splitext(file)
    match ext.lower():
        case ".json":
            labels, levels = labels_from_json(file)
        case ".csv":
            labels, levels = labels_from_csv(file)
        case _:
            labels, levels = labels_from_txt(file)
    labels = sorted(labels, key=lambda x: x[::-1])
    levels = len(labels[0])
    colnames = TAXONOMY_KEYS[:levels]
    if output is not None:
        with open(output, "w", encoding="utf8") as f:
            writer = csv.writer(f)
            writer.writerow(colnames)
            for label in labels:
                writer.writerow(label)
    return labels


def cli():
    parser = ArgumentParser(
        prog="class2combinations",
        description="Create a hierarchical combinations file from a list of species or a class specification created by mini_trainer",
    )
    parser.add_argument(
        "-i",
        "--input",
        dest="file",
        type=str,
        required=True,
        help="Source for creating the combinations, "
        "either a file with a list of species, "
        "or a class specification JSON created by mini_trainer.",
    )
    parser.add_argument("-o", "--output", type=str, required=True, help="Output file.")

    return vars(parser.parse_args())


if __name__ == "__main__":
    main(**cli())
