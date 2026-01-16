import os
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pp
from tqdm import tqdm

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

def nrow(path : str):
    return sum(p.count_rows() for p in pp.ParquetDataset(path).fragments)


def iter_parquet(path : str, columns=COLUMNS):
    """Iterate lazily over rows in ``gbifxdl`` parquet.
    """
    for batch in pp.ParquetFile(path).iter_batches(columns=columns):
        yield from batch.filter(
            pc.match_substring_regex(pc.field("set"), pattern="^\\d+$")
        ).select(
            columns
        ).tolist()


def iter_parquet_batches(path : str, columns=COLUMNS):
    """Iterate lazily over rows in ``gbifxdl`` parquet.
    """
    for batch in pp.ParquetFile(path).iter_batches(columns=columns):
        filtered = batch.filter(
            pc.match_substring_regex(pc.field("set"), pattern="^\\d+$")
        ).select(
            columns
        )
        if filtered.num_rows > 0:
            yield filtered


def set2split(set : int):
    """Map: 0=test, 1=validation, *=train.
    """
    match set:
        case 0:
            return "test"
        case 1:
            return "validation"
        case _:
            return "train"


def path_from_class(file : str, gid : int, dir : str):
    """Compose full path from basename, class, and root directory.
    """
    return os.path.join(dir, "images", str(gid), file)


def get_keys(row : dict[str, Any]):
    """Get class values from ``gbifxdl` row.
    """
    return [str(int(row[k].strip())) for k in KCOLUMNS]


def combine_dicts(dicts : Iterable[dict]):
    """Combine dictionaries with shared keys by stacking values in lists.
    """
    if not isinstance(dicts, (list, tuple)):
        dicts = list(dicts)
    if len(dicts) == 0:
        return dict()
    retval = {k : [] for k in dicts[0].keys()}
    for d in dicts:
        for k, v in d.items():
            retval[k].append(v)
    return retval


# def get_metadata_from_parquet(
#         path : str, 
#         cls2idx : dict[str, int | dict[str, int]],
#         **kwargs
#     ) -> dict[Literal['split', 'class', 'path', 'label'], list[str | int]]:
#     """This functions retrieves the metadata index for use with minitrainer.
    
#     Args:
#         path: Path to parquet created by ``gbifxdl``.
#         cls2idx: A dictionary with mappings from GBIF taxon (probably species) IDs to indexes used for DL training.
#             Can also be a dictionary with mappings from ``"0"``-``"N"`` to dictionaries as described above, 
#             where the key denotes the taxonomic level, such that ``"0"`` is species, ``"1"`` is genus and so forth.
#         kwargs: unused.
#     """
#     if isinstance(cls2idx[next(iter(cls2idx))], dict):
#         def parse_row(row : dict[str, Any]):
#             nonlocal path
#             split = set2split(int(row["set"].strip()))
#             keys = get_keys(row)
#             cls : list[int] = [cls2idx[str(level)][keys[level]] for level in range(len(cls2idx))]
#             filepath = path_from_class(file=row["filename"], gid=keys[0], dir=os.path.dirname(os.path.abspath(path)))
#             return {"split" : split, "class" : cls, "path" : filepath, "label" : keys}
#     else:
#         def parse_row(row : dict[str, Any]):
#             nonlocal path
#             split = set2split(int(row["set"].strip()))
#             keys = get_keys(row)
#             cls : int = cls2idx[keys[0]]
#             filepath = path_from_class(file=row["filename"], gid=keys[0], dir=os.path.dirname(os.path.abspath(path)))
#             return {"split" : split, "class" : cls, "path" : filepath, "label" : keys[0]}
    
#     return combine_dicts(map(parse_row, tqdm(iter_parquet(path), desc=f"Parsing metadata from {path}...", total=nrow(path))))

def get_metadata_from_parquet(
        path: str, 
        cls2idx: dict[str, int | dict[str, int]], 
        **kwargs
    ) -> dict[str, list]:
    
    root_dir = os.path.dirname(os.path.abspath(path))
    is_hierarchical = isinstance(cls2idx[next(iter(cls2idx))], dict)
    
    # Pre-calculate string constants for path construction to speed up loop
    # We will use vectorized string concatenation
    base_dir = os.path.join(root_dir, "images") + os.sep

    results = []
    
    # We manually manage the progress bar to count ROWS, not batches
    with tqdm(total=nrow(path), desc=f"Parsing metadata (Vectorized) from {path[-min(25, len(path)):]}...") as pbar:
        
        for batch in iter_parquet_batches(path):
            # 1. Convert Arrow Batch to Pandas DataFrame (Zero-copy where possible)
            df = batch.to_pandas()
            
            # 2. Vectorized Path Construction
            # Logic: dir + "images/" + speciesKey + "/" + filename
            # We assume 'speciesKey' corresponds to gid (keys[0]) as per your original logic
            # Converting series to string might be needed if they are ints
            gid_col = df[KCOLUMNS[0]].astype(str)
            df['path'] = base_dir + gid_col + os.sep + df['filename']

            # 3. Vectorized Split Mapping
            # Logic: 0->test, 1->val, else->train
            # We use numpy select for fast conditional logic
            sets = pd.to_numeric(df['set'], errors='coerce').fillna(-1)
            conditions = [sets == 0, sets == 1]
            choices = ["test", "validation"]
            df['split'] = np.select(conditions, choices, default="train")

            # 4. Vectorized Class/Label Mapping
            if is_hierarchical:
                # Handle hierarchical (list of classes)
                # We create a matrix of classes
                class_cols = []
                label_cols = []
                
                for level in range(len(cls2idx)):
                    key_col_name = KCOLUMNS[level]
                    # Map the column using the dictionary for this level
                    # map() is much faster than a loop comprehension
                    col_str = df[key_col_name].astype(str).str.strip()
                    mapped_cls = col_str.map(cls2idx[str(level)])
                    
                    class_cols.append(mapped_cls)
                    label_cols.append(col_str)
                
                # Stack results into lists (this is the only slow-ish part, but still faster)
                df['class'] = list(zip(*class_cols))
                df['label'] = list(zip(*label_cols))
                
            else:
                # Handle flat (single class)
                key_col_name = KCOLUMNS[0]
                col_str = df[key_col_name].astype(str).str.strip()
                
                df['class'] = col_str.map(cls2idx)
                df['label'] = col_str

            # 5. append valid columns to results
            # We only keep the columns your interface expects
            results.append(df[['split', 'class', 'path', 'label']])
            
            # Update progress bar by the actual number of rows processed
            pbar.update(len(df))

    # Concatenate all mini-dataframes and convert to dict of lists
    final_df = pd.concat(results, ignore_index=True)
    return final_df.to_dict(orient='list')


def parquet_to_class_spec(path : str):
    """Create flat class specification from ``gbifxdl`` parquet.
    """
    clss = set([row["speciesKey"].strip() for row in iter_parquet(path, ("speciesKey", ))])
    return {
        "cls2idx" : {cls : i for i, cls in enumerate(clss)},
        "num_classes" : len(clss)
    }


def parquet_to_class_spec_hierarchical(path : str, levels : int=3):
    """Create hierarchical class specification from ``gbifxdl`` parquet.
    """
    combs = {
        v[0] : v 
        for v in sorted(
            set([tuple(get_keys(row)[:levels]) for row in iter_parquet(path, KCOLUMNS)]), 
            key=lambda x : x[::-1]
        )
    }
    cls2idx = dict()
    for level in range(levels):
        clss = set()
        this_cls2idx = dict()
        for _, comb in combs.items():
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
