# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "diskcache",
#     "tqdm",
#     "pillow"
# ]
# ///

import csv
import hashlib
import json
import math
import os
import re
from argparse import ArgumentParser
from collections import Counter, OrderedDict
from dataclasses import asdict, dataclass, fields
from types import NoneType
from typing import get_args
from urllib.parse import quote
from urllib.request import urlopen

from diskcache import Cache
from PIL import Image
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import thread_map

## GBIF API Handling
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


@cache.memoize(expire=7 * 86400) # One week
def retrive_request(req : str):
    """Retrieve a composed HTTPS request.
    """
    if not req.startswith("https://"):
        raise NotImplementedError("Only HTTPS APIs are currently supported.")
    with urlopen(req) as resp:
        if resp.status != 200:
            raise RuntimeError(f'Unable to resolve request, received status {resp.status} from {req}.')
        return json.load(resp)


@dataclass(kw_only=True)
class GBIFTaxa:
    """Convenience container for GBIF taxa.
    """
    species_name : str | None
    species_id : int | None
    genus_name : str | None
    genus_id : int | None
    family_name : str | None
    family_id : int | None
    order_name : str | None
    order_id : int | None
    class_name : str | None
    class_id : int | None
    phylum_name : str | None
    phylum_id : int | None
    kingdom_name : str | None
    kingdom_id : int | None

    @classmethod
    def from_kwargs(cls, **kwargs):
        proc = {}
        for rank, value in kwargs.items():
            rank = rank.removesuffix("_")
            if rank.split("_")[-1] in ("id", "name"):
                proc[rank] = value
            else:
                id, name = value
                idk, namek = f'{rank}_id', f'{rank}_name'
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
        
        def _full_rank(rank : str):
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
        fmt = "\n  ".join([f'{r.title():>7}: {{{r}}}' for r in ranks])
        fmt = f'GBIFTaxa(\n  {fmt}\n)'
        data = {
            level : f'{getattr(self, level + "_name")} [{getattr(self, level + "_id")}]'
            for level in ranks
        }
        return fmt.format(**data)


def resolve_id(id : str | int):
    """Resolves a GBIF id to the accepted GBIF id and scientific name for all taxonomic levels.
    
    * `[species, genus, family, order, class, phylum, kingdom]` 
    
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


SPACE_PATTERN = re.compile(r'\s[x×]\s|[\s_]+')


def parse_name(name : str | None, user_author : str | None=None):
    """Parse taxa name and author from scientific name-string.
    """
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
        threshold : int=0,
        attempt : int=0,
        max_attempts : int=10
    ) -> tuple[int, str, int]:
    """Convert taxa name to GBIF ID.
    
    Returns:
        (key, rank, confidence): Returns the matched GBIF `usageKey` and `rank`, and the matching confidence.
    """
    attempt += 1
    if attempt > max_attempts:
        raise RuntimeError(f'Unable to convert {name} ({author=}, {rank_contains=}) at {threshold=} to GBIF id in {max_attempts=}')
    name, _ = parse_name(name, author)
    try:
        req = f'{GBIF_SPECIES_API_ENDPOINT}match?name={quote(name)}'
        if author is not None:
            req = f'{req}&authorship={quote(author)}'
        data = retrive_request(req)
        id, rank, conf = (data.get(k, None) for k in ["usageKey", "rank", "confidence"])
        if rank == "GENUS" and conf >= threshold:
            return name_to_id(
                " ".join([data["genus"], name.split(" ")[1]]),
                rank_contains=rank_contains,
                threshold=threshold,
                attempt=attempt,
                max_attempts=max_attempts
            )
        if (
            not (isinstance(id, int) and isinstance(rank, str) and isinstance(conf, int)) or 
            (rank_contains is not None and rank_contains not in rank) or 
            conf < threshold
        ):
            raise RuntimeError(f'Unable to properly resolve {name} using "{req}" got {id=} {rank=} {conf=}:\n{data}') 
        return id, rank, conf
    except Exception as e:
        if "Unable to convert" in str(e):
            raise e
        req = f'{GBIF_SPECIES_API_ENDPOINT}search?nameType=SCIENTIFIC&q={quote(name)}'
        data = retrive_request(req)["results"]
        if len(data) == 0 or (new_name := parse_name(data[0].get("scientificName", None))[0]) is None:
            e.add_note(f'Request: {req}')
            raise e
        if (
            name == new_name and 
            (id := data[0]["speciesKey"]) and 
            (rank_contains is not None and rank_contains in (rank := data[0].get("rank", "UNKNOWN")))
        ):
            if isinstance(id, str):
                id = id.strip()
                if id.isdigit():
                    id = int(id)
            assert isinstance(id, int)
            assert isinstance(rank, str)
            return id, rank, threshold
        return name_to_id(new_name, rank_contains=rank_contains, threshold=threshold, attempt=attempt, max_attempts=max_attempts)

############################################


## Various helper functions
def folder_size(directory):
    def _folder_size(directory):
        total = 0
        for entry in os.scandir(directory):
            if entry.is_dir():
                _folder_size(entry.path)
                total += parent_size[entry.path]
            else:
                size = entry.stat().st_size
                total += size
                file_size[entry.path] = size
        parent_size[directory] = total

    file_size = {}
    parent_size = {}
    _folder_size(directory)
    return file_size, parent_size


def listdir_recursive(path : str):
    out = []
    if os.path.isfile(path):
        out.append(path)
    elif os.path.isdir(path):
        for nxt in os.listdir(path):
            out.extend(listdir_recursive(os.path.join(path, nxt)))
    return out


def format_bytes(size):
    power = 0 if size <= 0 else math.floor(math.log(size, 1024))
    return f"{round(size / 1024 ** power, 2)} {['B', 'KB', 'MB', 'GB', 'TB'][int(power)]}"


def validate(root : str, image_dir : str, metadata : str, taxonomy : str, verbose : bool=True):
    """Attempt to validate if the dataset is complete, may not work on Windows."""
    VALIDATION_RESULT = []
    VALIDATION_HASH = 252552719891186879393839677190393325384

    headers = f'| {"Type ":^7} | {"Content size":^15} | {root + os.sep + "*":<35} |'
    hline = "-" * len(headers)
    if verbose:
        print(headers)
        print(hline)
    for f in [image_dir, metadata, taxonomy]:
        if not os.path.exists(f):
            ftype = "X"
            size = 0
        elif os.path.isdir(f):
            ftype = "D"
            size = folder_size(f)[1][f]
        elif os.path.isfile(f):
            ftype = "F"
            size = os.path.getsize(f)
        else:
            ftype = "?"
            size = -1
        VALIDATION_RESULT.append((f, ftype, size))
        if verbose:
            print(f'| {ftype:^7} | {format_bytes(size):>15} | {os.path.relpath(f, root):>35} |')
    if verbose:
        print(hline)

    chk = hashlib.md5(usedforsecurity=False)
    [chk.update(str(v).encode()) for e in VALIDATION_RESULT for v in e]
    chk = int.from_bytes(chk.digest(), "big")
    if chk != VALIDATION_HASH:
        print(
            'Incorrect PlantNet300K V2 content validation hash:\n'
            f'\t|    found: {chk} |\n'
            f'\t| expected: {VALIDATION_HASH} |'
        )
    else:
        if verbose:
            print("Contents match </")


@dataclass(frozen=True)
class PlantNet300KException:
    """Simple regex-based conversion-rule."""
    pattern : str | re.Pattern
    repl : str
    fields : list[str]

    def apply(self, other : "PlantNet300KClass"):
        """Return a new copy of the PlantNet300KClass, where matches of `pattern` in `fields` are replaced with `repl`."""
        data = other.to_dict()
        new = data.copy()
        for field in self.fields:
            new[field] = re.sub(self.pattern, self.repl, data[field])
        return type(other)(**new)


@dataclass
class PlantNet300KClass:
    """Struct for PlantNet300K species/class metadata with converter to GBIF/mini_trainer."""
    species_id : int
    full_species : str
    species : str
    genus : str
    family : str
    epithet : str
    author : str
    unmatched_terms : str
    iucn_status : str

    def to_gbif(self):
        """Attempt to resolve self to the corresponding canonical (accepted) species ID in the GBIF Backbone."""
        key, rank, conf = name_to_id(self.species, self.author)
        if rank != "SPECIES":
            raise RuntimeError(f'Attempted to convert {self} to GBIF but found {key} of rank: {rank}, expected: SPECIES')
        return GBIFTaxa.from_kwargs(**resolve_id(key))

    def to_dict(self):
        return asdict(self)

 
EXCEPTIONS = [
    PlantNet300KException("Oreomecon", "Papaver", ["full_species", "species", "genus"])
] # Conversion exceptions/rules for classes in the species metadata of the Pl@ntNet300K V2 dataset


def barchart(x, max_width : int=50, transform=None):
    rcs = Counter(x)
    rccs = Counter(rcs.values())
    ircs = {v : k for k, v in rcs.items() if rccs[v] == 1}
    cs, fs = zip(*sorted(rccs.items()))
    if transform is None:
        fns = fs
    else:
        fns = [transform(f) for f in fs]
    mf = max(fns)
    cmw = len(str(max(cs)))
    max_width = max_width - cmw - 3
    return "\n".join([
            f'{c:>{cmw}} : {"#" * math.floor(max_width * fn / mf):|>1} ({f if f != 1 else ircs[c]})' 
            for c, f, fn in zip(cs, fs, fns)
    ])


def rewrite_image_pillow(src: str, dst: str, size : int):
    if os.path.exists(dst):
        return -1
    image = Image.open(src).convert("RGB")
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    image.save(dst, "JPEG", quality=95)
    return 1


def resize_images(src : list[str], dst : list[str], size : int, verbose : bool=True):
    assert len(src) == len(dst)
    def _resize_one(s_d):
        s, d = s_d
        os.makedirs(os.path.dirname(d), exist_ok=True)
        return rewrite_image_pillow(s, d, size)
    result = Counter(thread_map(
        _resize_one, 
        zip(src, dst), 
        tqdm_class=tqdm,
        total=len(src),
        max_workers=min(64, max(1, os.cpu_count() // 2)),
        chunksize=32,
        leave=verbose,
        dynamic_ncols=True
    ))
    existing, resized = result.get(-1, 0), result.get(1, 0)
    return existing, resized


def normalize_split(split : str, options : tuple[str, ...]=("train", "validation", "test")):
    split = split.lower().strip()
    for opt in options:
        if split in opt:
            return opt
    raise ValueError(f"Unknown split: '{split}', expected one of [{", ".join(f"'{opt}'" for opt in options)}]")

####################################


## Program flow
def main(verbose : bool=True):
    NSTEPS = 8
    with tqdm(total=NSTEPS, unit="step", leave=verbose, dynamic_ncols=True) as pbar:
        ROOT = os.getcwd()
        IMAGE_DIR = os.path.join(ROOT, "images")
        METADATA = os.path.join(ROOT, "plantnet300K_metadata.csv")
        TAXONOMY = os.path.join(ROOT, "species_metadata.csv")

        # 1) Check dataset integrity
        pbar.set_description_str("Validating...")
        validate(root=ROOT, image_dir=IMAGE_DIR, metadata=METADATA, taxonomy=TAXONOMY, verbose=verbose)
        pbar.update(1)

        # 2) Read species metadata
        pbar.set_description_str("Processing species/class metadata...")
        with open(TAXONOMY) as f:
            tax = {c[0] : c[1:] for c in zip(*list(csv.reader(f)))}
        pbar.update(1)

        # 3) Parse Pl@ntNet300K V2 class metadata into standardized struct
        pbar.set_description_str("Parsing PlantNet300K and applying exceptions...")
        cols = list(tax.keys())
        tax_ncls = len(tax[list(tax)[0]])
        pln_sps, pln_skip = [], []
        for row in tqdm(zip(*(tax[c] for c in cols)), total=tax_ncls, leave=False, dynamic_ncols=True):
            row = {c : v for c, v in zip(cols, row)}
            sp = PlantNet300KClass(**row)
            # Apply conversion exceptions/rules
            for ex in EXCEPTIONS:
                sp = ex.apply(sp)
            # Skip genus-level classes
            if sp.species.split(" ")[-1].strip().lower() in ["sp", "sp.", "spp", "spp."]:
                pln_skip.append(sp)
            else:
                pln_sps.append(sp)
        pl_ncls = len(pln_sps)
        pbar.update(1)

        # 4) Resolve classes to canonical (accepted) GBIF species ID
        # (we collapse classes that resolve to the same GBIF ID)
        pbar.set_description_str("Resolving PlantNet300K species to GBIF IDs")
        gbif_sps : list[GBIFTaxa] = []
        gbif_collapsed : dict[GBIFTaxa, list[PlantNet300KClass]] = dict()
        skipped = 0
        for src_sp in tqdm(pln_sps, leave=False, dynamic_ncols=True):
            sp = src_sp.to_gbif()
            if sp not in gbif_collapsed:
                gbif_collapsed[sp] = [src_sp]
                gbif_sps.append(sp)
            else:
                skipped += 1
                gbif_collapsed[sp].append(src_sp)
        gbif_sps = sorted(gbif_sps, key=lambda x : [getattr(x, rank + "_name") for rank in x.ranks[::-1]], reverse=False)
        gbif_ncls = len(gbif_sps)
        gbif_skip = "\n".join([
            f"  {k.species_name} [{k.species_id}]:\n    - " + "\n    - ".join(map(repr, v)) 
            for k, v in gbif_collapsed.items() if len(v) > 1
        ])
        if verbose:
            print(f"Parsed {tax_ncls} Pl@ntNet300K V2 classes")
            print(f"Found {pl_ncls} valid species-level classes, these were removed:")
            print("  " + "\n  ".join("\n".join(map(repr, pln_skip)).split("\n")))
            print(f"Resolved {gbif_ncls} classes to a unique GBIF species ID, these were duplicates:")
            print(gbif_skip)
            print()

            for rank in ["species", "genus", "family", "order", "class", "phylum", "kingdom"]:
                print(f"Barchart[{rank.title()}]")
                print(barchart([getattr(r, rank + "_name") for r in gbif_sps], max_width=80, transform=None))
                print()
            
            print("All classes/species processed.\n")
        pbar.update(1)

        # 5) Parent directory for standardized dataset
        pbar.set_description_str("Constructing standardized dataset...")
        NEW_IMAGE_DIR = os.path.join(ROOT, "images_gbif")

        # Read dataset metadata
        with open(METADATA) as f:
            meta = {c[0] : c[1:] for c in zip(*list(csv.reader(f)))}
        
        # Construct expected original path
        meta["path"] = [
            os.path.join(IMAGE_DIR, split, f'{id:0>4}', hsh + ".jpg") 
            for split, id, hsh in zip(meta["split"], meta["species_id"], meta["PN_hash"])
        ]
        # Construct new path in the standardized dataset
        plid_to_gbifid = dict()
        for gbif_sp, tpl_sps in gbif_collapsed.items():
            for pl_sp in tpl_sps:
                plid_to_gbifid[pl_sp.species_id] = gbif_sp.species_id
        meta["gbif_id"] = [plid_to_gbifid.get(id, None) for id in meta["species_id"]]
        meta_orig = meta.copy()
        mask = [i for i, gid in enumerate(meta["gbif_id"]) if gid is not None]
        meta = {k : [v[i] for i in mask] for k, v in meta.items()}
        # OBS: Notice the original path is moved to the `orig_path` field
        meta["orig_path"] = meta["path"]
        # OBS: while the new path populates the `path` field 
        meta["path"] = [
            os.path.join(NEW_IMAGE_DIR, str(gid), hsh + ".jpg")
            for gid, hsh in zip(meta["gbif_id"], meta["PN_hash"])
        ]
        n_imgs_orig = len(meta_orig["path"])
        n_imgs = len(meta["path"])
        assert len(set(list(map(len, meta.values())))) == 1

        if verbose:
            print("Source Pl@ntNet300K V2 info:")
            print(f'Found {sum(map(os.path.exists, meta["orig_path"]))}/{n_imgs} images in {IMAGE_DIR}')
            print(f'Found {sum(map(os.path.exists, meta["path"]))}/{n_imgs} images in {NEW_IMAGE_DIR}')
            print("full - filtered = removed")
            print(n_imgs_orig, "-", n_imgs, "=", n_imgs_orig - n_imgs)
            print()
        pbar.update(1)

        # 6) Copy the source files in the original dataset to the destination in the standardized dataset
        pbar.set_description_str("Copying images...")
        existing, resized = resize_images(meta["orig_path"], meta["path"], size=512, verbose=verbose)
        if verbose:
            print(f"Finished resizing and moving {resized} images.\n")
        pbar.update(1)

        # 7)
        pbar.set_description_str("Removing unexpected files...")
        new_dataset = set(meta["path"])
        removed = 0
        for path in tqdm(listdir_recursive(NEW_IMAGE_DIR), leave=False, dynamic_ncols=True):
            if not os.path.isfile(path):
                continue
            if path not in new_dataset:
                os.remove(path)
                removed += 1
        if verbose:
            print(f"Finished removing {removed} unexpected files.")
            print("Finished constructing new dataset.")
            print("existing + resized - removed = total/expected")
            print(f'{existing} + {resized} - {removed} = {existing + resized}/{len(meta["path"])}')
        pbar.update(1)
        
        # 8) Store train-val-test and labels in a clean metadata json (data_index.json)
        pbar.set_description_str("Saving original train-val-test split metadata...")
        gid2hierarchy = {sp.species_id : sp.ids for sp in gbif_sps}
        NEW_METADATA = os.path.join(ROOT, "data_index.json")
        new_meta = {k : [] for k in ["path", "split", "label"]} # label is the *name* of the class
        new_meta["path"] = [os.path.relpath(path, ROOT) for path in meta["path"]] # Just store the relative path
        new_meta["split"] = [normalize_split(spl) for spl in meta["split"]]
        new_meta["label"] = [list(map(str, gid2hierarchy[gid])) for gid in meta["gbif_id"]]
        # Filter non-variable label levels 
        # (i.e. truncate the labels to the highest taxa rank where we have more than 1 taxa)
        label_nunique = [len(set(labs)) for labs in zip(*new_meta["label"])]
        label_levels = min(
            (i for i, v in enumerate(label_nunique) if v == 1), 
            default=len(label_nunique)
        )
        if verbose:
            print(f"Outputting {label_levels} out of {len(label_nunique)} label levels")
        new_meta["label"] = [labs[:label_levels] for labs in new_meta["label"]]
        
        if os.path.exists(NEW_METADATA):
            os.remove(NEW_METADATA)
            if verbose:
                print(f"Removing old metadata (data index): {NEW_METADATA}")
        with open(NEW_METADATA, "w") as f:
            json.dump(new_meta, f)
        if verbose:
            print(f"Created metadata (data index): {NEW_METADATA}")
        pbar.update(1)

    if verbose:
        print("Done!")


if __name__ == "__main__":
    parser = ArgumentParser(
        "format-plantnet300k", 
        description=(
            "Standardize the Pl@ntNet300K V2 dataset (10.5281/zenodo.10419064) to species level in the GBIF backbone. "
            "Classes are resolved to their canonical (accepted) species ID via the GBIF API. "
            "Please not that classes which are not defined at the species level (e.g. 'XXX spp.') are removed and "
            "classes which resolve to the same GBIF species ID are collapsed."
        )
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", default=False, required=False
    )
    args = parser.parse_args()
    main(**vars(args))
