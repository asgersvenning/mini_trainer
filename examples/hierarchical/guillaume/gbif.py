import json
from collections import OrderedDict
from urllib.request import urlopen

GBIF_SPECIES_API_ENDPOINT = 'https://api.gbif.org/v1/species/'
TAXONOMY_KEYS = [
    "kingdom",
    "phylum",
    "order",
    "family",
    "genus",
    "species"
]

def resolve_id(id : str | int):
    req = f'{GBIF_SPECIES_API_ENDPOINT}{id}'
    resp = urlopen(req)
    if resp.status != 200:
        raise RuntimeError(f'Unable to resolve GBIF species ID {id}, received status {resp.status} from {req}.')
    data = json.load(resp)
    clean_data = OrderedDict([(key, [data[key], data[f'{key}Key']]) for key in TAXONOMY_KEYS])
    return clean_data