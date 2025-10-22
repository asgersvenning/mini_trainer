import json
import os
from typing import Any, Iterable, Literal

import pyarrow.compute as pc
import pyarrow.parquet as pp

KCOLUMNS = (
    "speciesKey",
    "genusKey",
    "familyKey",
    "orderKey",
    "classKey",
    "phylumKey",
    "kingdomKey"
)

COLUMNS = (
    "filename",
    "set",
    *KCOLUMNS
)

def iter_parquet(path : str, columns=COLUMNS):
    for batch in pp.ParquetFile(path).iter_batches():
        for row in batch.filter(
                pc.match_substring_regex(pc.field("set"), pattern="^\\d+$")
            ).select(
                columns
            ).to_pylist():
            yield row
    
def set2split(set : int):
    if set == 0:
        return "test"
    if set == 1:
        return "validation"
    return "train"

def path_from_class(file : str, gid : int, dir : str):
    return os.path.join(dir, "images", str(gid), file)

def get_keys(row : dict[str, Any]):
    return [str(int(row[k].strip())) for k in KCOLUMNS]

def combine_dicts(dicts : Iterable[dict]):
    if not isinstance(dicts, (list, tuple)):
        dicts = list(dicts)
    if len(dicts) == 0:
        return dict()
    retval = {k : [] for k in dicts[0].keys()}
    for d in dicts:
        for k, v in d.items():
            retval[k].append(v)
    return retval

def get_metadata_from_parquet(
        path : str, 
        cls2idx : dict[str, int | dict[str, int]],
        **kwargs
    ) -> dict[Literal['split', 'class', 'path'], list[str | int]]:
    """
    This functions retrieves the metadata index for use with minitrainer.
    
    Args:
        path: Path to parquet created by ``gbifxdl``.
        cls2idx: A dictionary with mappings from GBIF taxon (probably species) IDs to indexes used for DL training.
            Can also be a dictionary with mappings from ``"0"``-``"N"`` to dictionaries as described above, 
            where the key denotes the taxonomic level, such that ``"0"`` is species, ``"1"`` is genus and so forth.
    """
    if isinstance(cls2idx[next(iter(cls2idx))], dict):
        def parse_row(row : dict[str, Any]):
            nonlocal path
            split = set2split(int(row["set"].strip()))
            keys = get_keys(row)
            cls : list[int] = [cls2idx[level][keys[level]] for level in range(len(cls2idx))]
            filepath = path_from_class(file=row["filename"], gid=keys[0], dir=os.path.dirname(os.path.abspath(path)))
            return {"split" : split, "class" : cls, "path" : filepath}
    else:
        def parse_row(row : dict[str, Any]):
            nonlocal path
            split = set2split(int(row["set"].strip()))
            keys = get_keys(row)
            cls : int = cls2idx[keys[0]]
            filepath = path_from_class(file=row["filename"], gid=keys[0], dir=os.path.dirname(os.path.abspath(path)))
            return {"split" : split, "class" : cls, "path" : filepath}
    
    return combine_dicts(map(parse_row, iter_parquet(path)))

def parquet_to_class_spec(path : str):
    clss = set([row["speciesKey"].strip() for row in iter_parquet(path, ("speciesKey", ))])
    return {
        "cls2idx" : {cls : i for i, cls in enumerate(clss)},
        "num_classes" : len(clss)
    }

def parquet_to_class_spec_hierarchical(path : str, levels : int=3):
    combs = list(sorted(set([tuple(get_keys(row)[:levels]) for row in iter_parquet(path, KCOLUMNS)]), key=lambda x : x[::-1]))
    cls2idx = dict()
    for level in range(levels):
        clss = set()
        this_cls2idx = dict()
        for comb in combs:
            cls = comb[level]
            if cls in clss:
                continue
            this_cls2idx[cls] = len(clss)
            clss.add(cls)
        cls2idx[str(level)] = this_cls2idx
    return {
        "cls2idx" : cls2idx,
        "labels" : combs,
        "num_classes" : len(cls2idx["0"])
    }
