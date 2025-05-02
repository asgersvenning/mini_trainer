import json
from collections import OrderedDict
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

def resolve_id(id : str | int):
    req = f'{GBIF_SPECIES_API_ENDPOINT}{id}'
    resp = urlopen(req)
    if resp.status != 200:
        raise RuntimeError(f'Unable to resolve GBIF species ID {id}, received status {resp.status} from {req}.')
    data = json.load(resp)
    clean_data = OrderedDict([(key, [str(data[f'{key}Key']), str(data[key])]) for key in TAXONOMY_KEYS])
    return clean_data