import json
import os
import re
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass, fields
from types import NoneType
from typing import Any, Literal, get_args, overload
from urllib.parse import quote
from urllib.request import urlopen

from diskcache import Cache

from mini_trainer.utils import filter_ordered_dict, multithread_vectorize

GBIF_SPECIES_API_ENDPOINT = "https://api.gbif.org/v1/species/"
TK = Literal["species", "genus", "family", "order", "class", "phylum", "kingdom"]
TAXONOMY_KEYS: tuple[TK, ...] = get_args(TK)

_CACHE = None
CACHE_TIME = 7 * 86400  # One week in seconds


def get_cache():
    global _CACHE
    if _CACHE is None:
        _CACHE = Cache(os.path.expanduser("~/.cache/nrs"))
    return _CACHE


def retrive_request(req: str) -> Any:
    """Retrieve a composed HTTPS request."""
    if not req.startswith("https://"):
        raise NotImplementedError("Only HTTPS APIs are currently supported.")
    cache = get_cache()
    ck = f"retrive_request:{req}"
    cached_result = cache.get(ck)
    if cached_result is not None:
        return cached_result

    with urlopen(req) as resp:
        if resp.status != 200:
            raise RuntimeError(f"Unable to resolve request, received status {resp.status} from {req}.")
        data = json.load(resp)
        cache.set(ck, data, expire=CACHE_TIME)
        return data


@dataclass(kw_only=True)
class GBIFTaxa:
    """Convenience container for GBIF taxa."""

    species_name: str | None
    species_id: int | None
    genus_name: str | None
    genus_id: int | None
    family_name: str | None
    family_id: int | None
    order_name: str | None
    order_id: int | None
    class_name: str | None
    class_id: int | None
    phylum_name: str | None
    phylum_id: int | None
    kingdom_name: str | None
    kingdom_id: int | None

    @classmethod
    def from_kwargs(cls, **kwargs):
        proc = {}
        for rank, value in kwargs.items():
            rank = rank.removesuffix("_")
            if rank.split("_")[-1] in ("id", "name"):
                proc[rank] = value
            else:
                id, name = value
                idk, namek = f"{rank}_id", f"{rank}_name"
                if idk not in proc:
                    proc[idk] = id
                if namek not in proc:
                    proc[namek] = name
        return cls(**proc)

    def __post_init__(self):
        struct = fields(self)
        for field in struct:
            value = getattr(self, field.name, None)
            if value is None:
                continue
            tp = [tp for tp in get_args(field.type) if tp != NoneType]
            assert len(tp) == 1
            tp = tp[0]
            if not isinstance(value, tp):
                setattr(self, field.name, tp(value))

    @property
    def ranks(self):
        default = ["species", "genus", "family", "order", "class", "phylum", "kingdom"]

        def _full_rank(rank: str):
            name = getattr(self, rank + "_name")
            id = getattr(self, rank + "_id")
            return name is not None and id is not None

        return list(filter(_full_rank, default))

    @property
    def names(self) -> list[str]:
        return [getattr(self, rank + "_name") for rank in self.ranks]

    @property
    def ids(self) -> list[int]:
        return [getattr(self, rank + "_id") for rank in self.ranks]

    @property
    def rank(self):
        return self.ranks[0]

    @property
    def id(self) -> int:
        return getattr(self, self.rank + "_id")

    @property
    def name(self) -> str:
        return getattr(self, self.rank + "_name")

    def __hash__(self):
        return self.id

    def __repr__(self):
        ranks = self.ranks
        fmt = "\n  ".join([f"{r.title():>7}: {{{r}}}" for r in ranks])
        fmt = f"{self.__class__.__name__}(\n  {fmt}\n)"
        data = {level: f"{getattr(self, level + '_name')} [{getattr(self, level + '_id')}]" for level in ranks}
        return fmt.format(**data)


def resolve_id(id: str | int):
    """Resolves a GBIF id to the accepted GBIF id and scientific name for all taxonomic levels.

    * `[species, genus, family, order, class, phylum, kingdom]`

    Args:
        id: GBIF species ID.

    Returns:
        (species taxonomy):
        The taxonomy of the species given by ``id`` as a dictionary:
        [str] <"taxa_level">: [tuple[int, str]] (<"Accepted GBIF id">, <"Accepted scientific name">)
    """
    req = f"{GBIF_SPECIES_API_ENDPOINT}{id}"
    data = retrive_request(req)
    try:
        clean_data = OrderedDict([(key, (str(data[f"{key}Key"]), str(data[key]))) for key in TAXONOMY_KEYS])
    except KeyError as e:
        e.add_note(f"Missing keys in: {data}")
        raise
    return clean_data


SPACE_PATTERN = re.compile(r"\s[x×]\s|[\s_]+")


@overload
def parse_name(name: str, user_author: str) -> tuple[str, str]: ...
@overload
def parse_name(name: str, user_author: None = None) -> tuple[str, str | None]: ...
@overload
def parse_name(name: None, user_author: str) -> tuple[None, str]: ...
@overload
def parse_name(name: None, user_author: None = None) -> tuple[None, None]: ...
def parse_name(name: str | None, user_author: str | None = None) -> tuple[str | None, str | None]:
    """Parse taxa name and author from scientific name-string."""
    if name is None:
        return name, user_author
    name = re.sub(SPACE_PATTERN, " ", name)
    parts = name.split(" ")
    if len(parts) <= 2:
        return name, user_author
    if user_author is not None:
        raise RuntimeError(f'Found author in name ("{name}") while an author ("{user_author}") was also passed.')
    name = " ".join(parts[:2])
    author = " ".join(parts[2:])
    return name, author


def name_to_id(
    name: str, author: str | None = None, rank_contains: str | None = None, threshold: int = 0, attempt: int = 0, max_attempts: int = 10
) -> tuple[int, str, int]:
    """Convert taxa name to GBIF ID.

    Returns:
        (key, rank, confidence): Returns the matched GBIF `usageKey` and `rank`, and the matching confidence.
    """
    attempt += 1
    if attempt > max_attempts:
        raise RuntimeError(f"Unable to convert {name} ({author=}, {rank_contains=}) at {threshold=} to GBIF id in {max_attempts=}")
    name, _ = parse_name(name, author)
    try:
        req = f"{GBIF_SPECIES_API_ENDPOINT}match?name={quote(name)}"
        if author is not None:
            req = f"{req}&authorship={quote(author)}"
        data = retrive_request(req)
        id, rank, conf = (data.get(k, None) for k in ["usageKey", "rank", "confidence"])
        if rank == "GENUS" and conf >= threshold:
            return name_to_id(
                " ".join([data["genus"], name.split(" ")[1]]),
                rank_contains=rank_contains,
                threshold=threshold,
                attempt=attempt,
                max_attempts=max_attempts,
            )
        if (
            not (isinstance(id, int) and isinstance(rank, str) and isinstance(conf, int))
            or (rank_contains is not None and rank_contains not in rank)
            or conf < threshold
        ):
            raise RuntimeError(f'Unable to properly resolve {name} using "{req}" got {id=} {rank=} {conf=}:\n{data}')
        return id, rank, conf
    except Exception as e:
        if "Unable to convert" in str(e):
            raise
        req = f"{GBIF_SPECIES_API_ENDPOINT}search?nameType=SCIENTIFIC&q={quote(name)}"
        data = retrive_request(req)["results"]
        if len(data) == 0 or (new_name := parse_name(data[0].get("scientificName", None))[0]) is None:
            raise RuntimeError(f"Request {req}, returned empty, partial or malformed data: {data}") from e
        if (
            name == new_name
            and (id := data[0]["speciesKey"])
            and (rank_contains is not None and rank_contains in (rank := data[0].get("rank", "UNKNOWN")))
        ):
            if isinstance(id, str):
                id = id.strip()
                if id.isdigit():
                    id = int(id)
            assert isinstance(id, int)
            assert isinstance(rank, str)
            return id, rank, threshold
        return name_to_id(name=new_name, rank_contains=rank_contains, threshold=threshold, attempt=attempt, max_attempts=max_attempts)


@multithread_vectorize(desc="Translating names...")
def id_to_name(id: str | int):
    if isinstance(id, str):
        id = id.strip()
        if not id.isdigit():
            raise ValueError(f"{id} must be a digit.")
        id = int(id)
    req = f"{GBIF_SPECIES_API_ENDPOINT}{id}/name"
    data = retrive_request(req)
    return data["scientificName"]


@multithread_vectorize(desc="Resolving taxa...")
def resolve_name_or_id(name_or_id: str | int):  # noqa: D103
    if isinstance(name_or_id, str):
        name_or_id = name_or_id.strip()
        if name_or_id.isdigit():
            name_or_id = int(name_or_id)

    if isinstance(name_or_id, int):
        return resolve_id(name_or_id)

    id, rank, conf = name_to_id(name_or_id, rank_contains="SPECIES", threshold=90)
    return resolve_id(id)


def resolve_level(level: int | str):
    if isinstance(level, int):
        return TAXONOMY_KEYS[level]
    level = level.strip().lower()
    assert level in TAXONOMY_KEYS
    return level


def select_levels(levels: str | int | Iterable[str | int] | None, taxonomy: list[OrderedDict[TK, tuple[str, str]]]) -> list[TK]:
    # If levels is not None we consider the following cases:
    if levels is None:
        level_classes = OrderedDict((k, set(tax[k] for tax in taxonomy)) for k in TAXONOMY_KEYS)
        return [k for k, v in level_classes.items() if len(v) > 1]
    if isinstance(levels, (str, int)):
        return [TAXONOMY_KEYS[i] for i in range(TAXONOMY_KEYS.index(resolve_level(levels)) + 1)]
    return sorted([resolve_level(lvl) for lvl in levels], key=TAXONOMY_KEYS.index)


def create_taxonomy(  # noqa: D103
    names_or_ids: Iterable[str], levels: str | int | Iterable[str | int] | None = None
):
    taxs = resolve_name_or_id(names_or_ids)
    level_strs: list[TK] = select_levels(levels, taxs)
    return OrderedDict(
        [
            (orig, filter_ordered_dict(tax, level_strs))
            for orig, tax in sorted(zip(names_or_ids, taxs), key=lambda x: [v[1] for v in x[1].values()])
        ]
    )


def labels_from_taxonomy(tax: OrderedDict[str, OrderedDict[str, tuple[str, str]]]):  # noqa: D103
    return OrderedDict([(k, tuple([v[0] for v in e.values()])) for k, e in tax.items()])


def cls2idx_from_labels(labels: OrderedDict[str, tuple[str, ...]]):  # noqa: D103
    nlvl = set([len(lab) for lab in labels.values()])
    if len(nlvl) != 1:
        raise RuntimeError("Varying hierarchy levels found in image directory structure:", list(sorted(nlvl)))
    nlvl = list(nlvl)[0]
    cls2idx: dict[str, dict[str, int]] = {str(lvl): dict() for lvl in range(nlvl)}
    classes = {str(lvl): set() for lvl in range(nlvl)}
    for lab in labels.values():
        for lvl, cls in enumerate(lab):
            if cls in classes[str(lvl)]:
                continue
            classes[str(lvl)].add(cls)
            cls2idx[str(lvl)][cls] = len(classes[str(lvl)]) - 1
    return cls2idx


def _keys_str_int(d: dict):
    return all([isinstance(k, str) and k.isdigit() for k in d.keys()])


def _values_int(d: dict):
    return all([isinstance(v, int) for v in d.values()])


def is_taxonomical_cls2idx(cls2idx):  # noqa: D103
    if not isinstance(cls2idx, dict):
        return False
    if len(cls2idx) == 0:
        return False
    if not _keys_str_int(cls2idx):
        return False
    return all([isinstance(v, dict) and _keys_str_int(v) and _values_int(v) for v in cls2idx.values()])
