import os, json
from pyremotedata.implicit_mount import IOHandler
from hierarchical.base.gbif import resolve_id, name_to_id, TAXONOMY_KEYS
from typing import Union
from tqdm.contrib.concurrent import thread_map

from collections import OrderedDict

def erda_list_files(id : str, **kwargs):
    with IOHandler(user=id, password=id, remote="io.erda.au.dk") as io:
        return io.get_file_index()

def resolve_name_or_id(name_or_id : Union[str, int]):
    name_or_id = name_or_id.strip()
    if isinstance(name_or_id, int) or name_or_id.isdigit():
        return resolve_id(name_or_id)
    id, rank, conf = name_to_id(name_or_id, rank_contains="SPECIES", threshold=90)
    return resolve_id(id)

def create_taxonomy(names_or_ids : list[int], level : str="family"):
    level = level.strip().lower()
    if level not in TAXONOMY_KEYS:
        raise ValueError(f'Unknown taxonomic level "{level}", expected one of [{", ".join([f"\"{tk}\"" for tk in TAXONOMY_KEYS])}].')
    levels = TAXONOMY_KEYS[:(TAXONOMY_KEYS.index(level)+1)]
    del level
    info = {taxonomy["species"][0] : OrderedDict([(key, value) for key, value in taxonomy.items() if key in levels]) for taxonomy in thread_map(resolve_name_or_id, names_or_ids, total=len(names_or_ids), desc="Querying the GBIF Species API...")}
    return OrderedDict([(key, values) for key, values in sorted(info.items(), key=lambda x : [x[1][level][1] for level in levels])])

def names_or_ids_to_combinations(names_or_ids, highest_level : str="family", verbose : bool=False):
    taxonomy = create_taxonomy(names_or_ids, highest_level)
    if verbose:
        print(f'Created taxonomy of size {len(taxonomy)}:')
        for key, value in taxonomy.items():
            print(f'{key:<9}: {", ".join(f"{level}: {value[level]}" for level in value)}')

    taxonomy_store = "taxonomy_store.json"
    if os.path.exists(taxonomy_store):
        os.remove(taxonomy_store)
    with open(taxonomy_store, "w") as f:
        json.dump(taxonomy, f)
    if verbose:
        print("Taxonomy stored in:", taxonomy_store)

    combinations = [[value[level][0] for level in value] for value in taxonomy.values()]
    
    return combinations

# "JlICFo26h8"
def erda_to_combinations(id : str, verbose : bool=False):
    files = erda_list_files(id)
    if verbose:
        print(f'Found {len(files)} files.')
    folders = list(set(int(file.split("/")[0]) for file in files))
    if verbose:
        print(f'Found {len(folders)} folders.')
    return names_or_ids_to_combinations(folders, verbose=verbose)

