import json
import os
import re
from collections import OrderedDict
from typing import cast
from urllib.parse import quote
from urllib.request import urlopen

from diskcache import Cache

from mini_trainer.utils import filter_ordered_dict, multithread_vectorize

GBIF_SPECIES_API_ENDPOINT = 'https://api.gbif.org/v1/species/'
TAXONOMY_KEYS = (
    "species",
    "genus",
    "family",
    "order",
    "class",
    "phylum",
    "kingdom"
)

cache = Cache(os.path.expanduser('~/.cache/nrs'))

@cache.memoize(expire=7*86400)
def retrive_request(req : str):
    if not req.startswith("https://"):
        raise NotImplementedError("Only HTTPS APIs are currently supported.")
    with urlopen(req) as resp:
        if resp.status != 200:
            raise RuntimeError(f'Unable to resolve request, received status {resp.status} from {req}.')
        return json.load(resp)

def resolve_id(id : str | int):
    """
    Resolves a GBIF id to the accepted GBIF id and scientific name
    for all taxonomic levels:
    * ``[species, genus, family, order, class, phylum, kingdom]`` 
    
    Args:
        id: GBIF species ID.
    
    Returns:
        (species taxonomy): 
        The taxonomy of the species given by ``id`` as a dictionary: 
        [str] <"taxa_level">: [tuple[int, str]] (<"Accepted GBIF id">, <"Accepted scientific name">)
    """
    req = f'{GBIF_SPECIES_API_ENDPOINT}{id}'
    data = retrive_request(req)
    try:
        clean_data = OrderedDict([(key, (str(data[f'{key}Key']), str(data[key]))) for key in TAXONOMY_KEYS])
    except KeyError as e:
        e.add_note(f"Missing keys in: {data}")
        raise e
    return clean_data

SPACE_PATTERN = re.compile(r'[\s_]+')

def parse_name(name : str | None, user_author : str | None=None):
    if name is None:
        return name, user_author
    name = re.sub(SPACE_PATTERN, " ", name)
    parts = name.split(" ")
    if len(parts) == 2:
        return name, user_author
    if user_author is not None:
        raise RuntimeError(f'Found author in name ("{name}") while the an author ("{user_author}") was also passed.')
    name = " ".join(parts[:2])
    author = " ".join(parts[2:])
    return name, author

def name_to_id(
        name : str, 
        author : str | None=None, 
        rank_contains : str | None=None, 
        threshold : int=0
    ) -> tuple[int, str, int]:
    """
    Returns:
        (key, rank, confidence): Returns the matched GBIF `usageKey` and `rank`, and the matching confidence.
    """
    name, _ = parse_name(name, author)
    try:
        req = f'{GBIF_SPECIES_API_ENDPOINT}match?name={quote(name)}'
        if author is not None:
            req = f'{req}&authorship={quote(author)}'
        data = retrive_request(req)
        id, rank, conf = (data.get(k, None) for k in ["usageKey", "rank", "confidence"])
        if rank == "GENUS" and conf >= threshold:
            return name_to_id(" ".join([data["genus"], name.split(" ")[1]]), rank_contains=rank_contains, threshold=threshold)
        if not (isinstance(id, int) and isinstance(rank, str) and isinstance(conf, int)) or (rank_contains is not None and rank_contains not in rank) or conf < threshold:
            raise RuntimeError(f'Unable to properly resolve {name} using "{req}" got {id=} {rank=} {conf=}:\n{data}') 
        return id, rank, conf
    except Exception as e:
        req = f'{GBIF_SPECIES_API_ENDPOINT}search?nameType=SCIENTIFIC&q={quote(name)}'
        data = retrive_request(req)["results"]
        if len(data) == 0 or (new_name := parse_name(data[0].get("scientificName", None))[0]) is None:
            e.add_note(f'Request: {req}')
            raise e
        if name == new_name and (id := data[0]["speciesKey"]) and (rank_contains is not None and rank_contains in (rank := data[0].get("rank", "UNKNOWN"))):
            if isinstance(id, str):
                id = id.strip()
                if id.isdigit():
                    id = int(id)
            assert isinstance(id, int)
            assert isinstance(rank, str)
            return id, rank, threshold
        return name_to_id(new_name, rank_contains=rank_contains, threshold=threshold)
    
@multithread_vectorize(desc="Resolving taxa...")
def resolve_name_or_id(name_or_id : str | int):
    name_or_id = name_or_id.strip()
    if isinstance(name_or_id, int) or name_or_id.isdigit():
        return resolve_id(name_or_id)
    id, rank, conf = name_to_id(name_or_id, rank_contains="SPECIES", threshold=90)
    return resolve_id(id)

def create_taxonomy(names_or_ids : list[str], levels : str | int | tuple[str | int, ...] | list[str | int]="family"):
    # _levels = len(TAXONOMY_KEYS) - 1
    if isinstance(levels, str):
        _levels = TAXONOMY_KEYS.index(levels.strip().lower())
    elif isinstance(levels, (tuple, list)):
        _levels = [
            level if isinstance(level, int) else TAXONOMY_KEYS.index(level.strip().lower())
            for level in levels
        ]
    elif isinstance(levels, int):
        _levels = levels - 1
    else:
        raise TypeError(f'Unexpected type for {levels=} ({type(levels)}).')
    _levels = list(range(_levels + 1)) if isinstance(_levels, int) else _levels
    level_strs = [TAXONOMY_KEYS[lvl] for lvl in sorted(_levels)]
    taxs = resolve_name_or_id(names_or_ids)
    return OrderedDict([(orig, filter_ordered_dict(tax, level_strs)) for orig, tax in sorted(zip(names_or_ids, taxs), key=lambda x : [v[1] for v in x[1].values()])])

def labels_from_taxonomy(tax : OrderedDict[str, OrderedDict[str, tuple[str, ...]]]):
    return OrderedDict([(k, tuple([v[0] for v in e.values()])) for k, e in tax.items()])

def cls2idx_from_labels(labels : OrderedDict[str, tuple[str, ...]]):
    nlvl = set([len(l) for l in labels.values()])
    if len(nlvl) != 1:
        raise RuntimeError('Varying hierarchy levels found in image directory structure:', list(sorted(nlvl)))
    nlvl = list(nlvl)[0]
    cls2idx : dict[str, dict[str, int]] = {str(lvl) : dict() for lvl in range(nlvl)}
    classes = {str(lvl) : set() for lvl in range(nlvl)}
    for lab in labels.values():
        for lvl, cls in enumerate(lab):
            if cls in classes[str(lvl)]:
                continue
            classes[str(lvl)].add(cls)
            cls2idx[str(lvl)][cls] = len(classes[str(lvl)]) - 1
    return cls2idx

def _keys_str_int(d : dict):
    return all([isinstance(k, str) and k.isdigit() for k in d.keys()])
def _values_int(d : dict):
    return all([isinstance(v, int) for v in d.values()])

def is_taxonomical_cls2idx(cls2idx):
    if not isinstance(cls2idx, dict):
        return False
    if len(cls2idx) == 0:
        return False
    if not _keys_str_int(cls2idx):
        return False
    return all([isinstance(v, dict) and _keys_str_int(v) and _values_int(v) for v in cls2idx.values()])
    