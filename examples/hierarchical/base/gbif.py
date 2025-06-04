import json
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

@lru_cache(maxsize=1024)
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

def name_to_id(name : str, author : str | None=None) -> tuple[int, str, int]:
    """
    Returns:
        (key, rank, confidence) (tuple[int, str, int]): Returns the matched GBIF `usageKey` and `rank`, and the matching confidence.
    """
    req = f'{GBIF_SPECIES_API_ENDPOINT}match?name={quote(name)}'
    if author is not None:
        req = f'{req}&authorship={quote(author)}'
    data = retrive_request(req)
    return data["usageKey"], data["rank"], data["confidence"]