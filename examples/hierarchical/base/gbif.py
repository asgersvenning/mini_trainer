import json
import re
from collections import OrderedDict
from functools import lru_cache
from urllib.parse import quote
from urllib.request import urlopen

GBIF_SPECIES_API_ENDPOINT = 'https://api.gbif.org/v1/species/'
TAXONOMY_KEYS = (
    "species",
    "genus",
    "family",
    "order",
    "phylum",
    "kingdom"
)

@lru_cache(maxsize=2**15)
def retrive_request(req):
    resp = urlopen(req)
    if resp.status != 200:
        raise RuntimeError(f'Unable to resolve GBIF species ID {id}, received status {resp.status} from {req}.')
    return json.load(resp)

def resolve_id(id : str | int):
    req = f'{GBIF_SPECIES_API_ENDPOINT}{id}'
    data = retrive_request(req)
    clean_data = OrderedDict([(key, (str(data[f'{key}Key']), str(data[key]))) for key in TAXONOMY_KEYS])
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

def name_to_id(name : str, author : str | None=None, rank_contains : str | None=None, threshold : int=0):
    """
    Returns:
        (key, rank, confidence) (tuple[int, str, int]): Returns the matched GBIF `usageKey` and `rank`, and the matching confidence.
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
            return id, rank, threshold
        return name_to_id(new_name, rank_contains=rank_contains, threshold=threshold)