"""Sequential loading from indexed, uncompressed TAR shards (opt-in; an input directory holding ``manifest.json``).

A shard set is ``manifest.json`` plus, per shard, ``<name>.tar`` (uncompressed ustar, one member ``<key>.jpg`` per
sample, original encoded bytes) and ``<name>.parquet`` (one row per member in tar order: ``key``, ``offset`` of the
image bytes, ``size``, ``crc32``, and the manifest's ``label_columns``, finest rank first). Build sets with
``ShardWriter`` and check them with ``python -m mini_trainer.data.shards verify ROOT``.

Each epoch has one deterministic plan from ``(seed, epoch)``: shards are permuted, the concatenated sample stream is
cut into equal contiguous parts per rank, and each rank's part into per-worker parts of whole batches. Every sample
is therefore read at most once per epoch and resuming from an epoch boundary replays the same plan. Workers read
their shards sequentially and mix samples in a bounded buffer; the mixing across species comes from shuffling when
the set is built. Training drops only what equal batches on every rank require (fewer than ``world_size`` samples
plus one partial batch per rank); evaluation keeps every sample. Missing or corrupt data and labels outside the class
specification raise ``ShardError`` instead of skipping samples.
"""

import argparse
import hashlib
import json
import os
import tarfile
import zlib
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from mini_trainer import get_logger
from mini_trainer.utils import get_rank, get_world_size

FORMAT, VERSION = "mt-shards", 1
HEADER = 512
INDEX_COLUMNS = ("key", "offset", "size", "crc32")


class ShardError(RuntimeError):
    """A shard set is missing, inconsistent or corrupt."""


def _parquet():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as e:
        raise ImportError("Shard sets require the optional dependency: pyarrow. Install with `pip install mt-trainer[recommended]`.") from e
    return pa, pq


def is_shard_root(path: str | os.PathLike | None) -> bool:
    """Whether ``path`` is a directory holding a shard-set manifest."""
    return path is not None and os.path.isfile(os.path.join(path, "manifest.json"))


def load_manifest(root: str | os.PathLike) -> dict:
    """Read and validate ``root/manifest.json``."""
    path = os.path.join(root, "manifest.json")
    try:
        with open(path) as f:
            manifest = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise ShardError(f"Cannot read shard manifest {path}: {e}") from e
    if manifest.get("format") != FORMAT or manifest.get("version") != VERSION:
        raise ShardError(
            f"{path}: expected format {FORMAT!r} version {VERSION}, found {manifest.get('format')!r} {manifest.get('version')!r}."
        )
    for name, split in manifest["splits"].items():
        counted = sum(s["n_samples"] for s in split["shards"])
        if counted != split["n_samples"]:
            raise ShardError(f"{path}: split {name!r} declares {split['n_samples']} samples, its shards hold {counted}.")
    return manifest


def epoch_plan(
    counts: list[int],
    *,
    epoch: int,
    seed: int,
    shuffle: bool,
    rank: int,
    world: int,
    worker: int,
    num_workers: int,
    batch_size: int,
    drop_last: bool,
) -> list[tuple[int, int, int]]:
    """Return one worker's ``(shard, first, stop)`` segments for an epoch; see the module docstring."""
    first, stop = rank_range(sum(counts), rank, world, batch_size, drop_last)
    batches, partial = divmod(stop - first, batch_size)
    share, extra = divmod(batches, num_workers)
    begin = first + batch_size * (worker * share + min(worker, extra))
    end = begin + batch_size * (share + (worker < extra)) + (partial if worker == num_workers - 1 else 0)
    order = np.random.default_rng([seed, epoch]).permutation(len(counts)) if shuffle else np.arange(len(counts))
    segments, position = [], 0
    for shard in order.tolist():
        low, high = max(begin, position), min(end, position + counts[shard])
        if low < high:
            segments.append((shard, low - position, high - position))
        position += counts[shard]
    return segments


def rank_range(total: int, rank: int, world: int, batch_size: int, drop_last: bool) -> tuple[int, int]:
    """The ``[first, stop)`` positions of one rank in the epoch's sample stream."""
    if not drop_last:
        return rank * total // world, (rank + 1) * total // world
    per_rank = total // world
    return rank * per_rank, rank * per_rank + per_rank // batch_size * batch_size


class ShardDataset(IterableDataset):
    """Samples of one split, as ``(image, label)``, streamed from a shard set; see the module docstring.

    ``reader`` decodes encoded bytes; ``cls2idx`` is flat or per rank (``{"0": {...}, ...}``). With ``multilabel``, the
    label holds one class index per rank of ``cls2idx``.
    """

    def __init__(  # noqa: D107
        self,
        root: str | os.PathLike,
        split: str,
        reader: Callable[[bytes], torch.Tensor],
        cls2idx: dict,
        batch_size: int,
        *,
        train: bool,
        multilabel: bool = False,
        hook: Callable[[torch.Tensor], torch.Tensor] | None = None,
        seed: int = 0,
        shuffle_buffer: int = 2048,
        read_chunk: int = 16 * 2**20,
        verify_crc: bool = True,
    ):
        self.root, self.split = os.fspath(root), split
        manifest = load_manifest(root)
        if split not in manifest["splits"]:
            raise ShardError(f"Shard set {root} has no split {split!r}; it has {sorted(manifest['splits'])}.")
        self.shards = manifest["splits"][split]["shards"]
        self.counts = [s["n_samples"] for s in self.shards]
        self.label_columns = manifest["label_columns"]
        self.reader, self.hook, self.batch_size, self.multilabel = reader, hook, batch_size, multilabel
        self.train, self.seed, self.read_chunk, self.verify_crc = train, seed, read_chunk, verify_crc
        self.shuffle_buffer = shuffle_buffer if train else 0
        self.levels = [cls2idx[str(i)] for i in range(len(cls2idx))] if isinstance(next(iter(cls2idx.values())), dict) else [cls2idx]
        if len(self.levels) > len(self.label_columns):
            raise ShardError(f"The class specification has {len(self.levels)} ranks; {root} has labels for {len(self.label_columns)}.")
        self.rank, self.world, self.epoch = get_rank(), get_world_size(), 0
        self.total_samples = sum(self.counts)
        first, stop = rank_range(self.total_samples, self.rank, self.world, batch_size, train)
        self._length = stop - first
        if train and self.rank == 0:
            get_logger().info(
                f"Shards {split}: {self.total_samples:,} samples in {len(self.shards):,} shards; "
                f"{self._length * self.world:,} used per epoch "
                f"on {self.world} rank(s), {self.total_samples - self._length * self.world:,} dropped for equal batches."
            )

    def set_epoch(self, epoch: int) -> None:
        """Select the epoch whose plan the next iteration follows."""
        self.epoch = epoch

    def __len__(self) -> int:
        return self._length

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        info = get_worker_info()
        worker, num_workers = (info.id, info.num_workers) if info is not None else (0, 1)
        segments = epoch_plan(
            self.counts, epoch=self.epoch, seed=self.seed, shuffle=self.train, rank=self.rank, world=self.world,
            worker=worker, num_workers=num_workers, batch_size=self.batch_size, drop_last=self.train,
        )  # fmt: skip
        rng = np.random.default_rng([self.seed, self.epoch, self.rank, worker])
        if self.train:
            segments = [segments[i] for i in rng.permutation(len(segments))]
        expected, seen, buffer = sum(stop - first for _, first, stop in segments), 0, []
        for segment in segments:
            for sample in self._read_segment(*segment):
                if self.shuffle_buffer <= 1:
                    seen += 1
                    yield self._decode(sample)
                    continue
                buffer.append(sample)
                if len(buffer) >= self.shuffle_buffer:
                    seen += 1
                    yield self._decode(self._pop(buffer, rng))
        while buffer:
            seen += 1
            yield self._decode(self._pop(buffer, rng))
        if seen != expected:
            raise ShardError(f"Worker {worker} of rank {self.rank} produced {seen} samples, its plan holds {expected}.")

    @staticmethod
    def _pop(buffer: list, rng: np.random.Generator):
        i = int(rng.integers(len(buffer)))
        buffer[i], buffer[-1] = buffer[-1], buffer[i]
        return buffer.pop()

    def _decode(self, sample: tuple[bytes, np.ndarray, str, str]) -> tuple[torch.Tensor, torch.Tensor]:
        data, label, key, name = sample
        try:
            image = self.reader(data)
        except Exception as e:
            raise ShardError(f"Cannot decode sample {key!r} of shard {name}: {e}") from e
        if self.hook is not None:
            image = self.hook(image)
        return image, torch.as_tensor(label if self.multilabel else label[0], dtype=torch.long)

    def _index(self, position: int) -> dict:
        """Load and validate the sidecar index of one shard."""
        _, pq = _parquet()
        shard = self.shards[position]
        path = os.path.join(self.root, f"{shard['name']}.parquet")
        wanted = [*INDEX_COLUMNS, *self.label_columns[: len(self.levels)]]
        try:
            table = pq.read_table(path, columns=wanted)
        except Exception as e:
            raise ShardError(f"Cannot read shard index {path}: {e}") from e
        if table.num_rows != shard["n_samples"]:
            raise ShardError(f"{path}: {table.num_rows} rows, the manifest declares {shard['n_samples']}.")
        columns = table.to_pydict()
        labels = np.empty((table.num_rows, len(self.levels)), dtype=np.int64)
        for level, (column, mapping) in enumerate(zip(self.label_columns, self.levels)):
            try:
                labels[:, level] = [mapping[str(v)] for v in columns[column]]
            except KeyError as e:
                raise ShardError(f"{path}: label {e.args[0]!r} of {column} is not in the class specification.") from e
        return {**{k: columns[k] for k in INDEX_COLUMNS}, "labels": labels}

    def _open(self, position: int) -> tuple[int, str]:
        shard = self.shards[position]
        path = os.path.join(self.root, f"{shard['name']}.tar")
        try:
            fd = os.open(path, os.O_RDONLY)
        except OSError as e:
            raise ShardError(f"Cannot open shard {path}: {e}") from e
        size = os.fstat(fd).st_size
        if size != shard["tar_bytes"]:
            os.close(fd)
            raise ShardError(f"{path}: {size} bytes, the manifest declares {shard['tar_bytes']}.")
        return fd, path

    def _sample(self, view: memoryview, base: int, index: dict, i: int, path: str) -> bytes:
        """Return one sample's bytes from a chunk starting at file offset ``base``, after checking its header and CRC."""
        key, offset, size = index["key"][i], index["offset"][i], index["size"][i]
        start = offset - base
        header = bytes(view[start - HEADER : start])
        if (
            header[257:262] != b"ustar"
            or header[:100].split(b"\0", 1)[0].decode(errors="replace") != f"{key}.jpg"
            or int(header[124:136].split(b"\0", 1)[0].strip() or b"0", 8) != size
        ):
            raise ShardError(f"{path}: the tar header at byte {offset - HEADER} does not match sample {key!r}.")
        data = bytes(view[start : start + size])
        if self.verify_crc and zlib.crc32(data) != index["crc32"][i]:
            raise ShardError(f"{path}: CRC mismatch for sample {key!r} at byte {offset}.")
        return data

    def _read_segment(self, position: int, first: int, stop: int) -> Iterator[tuple[bytes, np.ndarray, str, str]]:
        index = self._index(position)
        offsets, sizes = index["offset"], index["size"]
        fd, path = self._open(position)
        try:
            i = first
            while i < stop:
                base, j = offsets[i] - HEADER, i + 1
                while j < stop and offsets[j] + sizes[j] - base <= self.read_chunk:
                    j += 1
                length = offsets[j - 1] + sizes[j - 1] - base
                try:
                    chunk = os.pread(fd, length, base)
                except OSError as e:
                    raise ShardError(f"Cannot read {length} bytes at {base} of {path}: {e}") from e
                if len(chunk) != length:
                    raise ShardError(f"{path}: short read at byte {base} ({len(chunk)} of {length}).")
                view = memoryview(chunk)
                for k in range(i, j):
                    yield self._sample(view, base, index, k, path), index["labels"][k], index["key"][k], path
                i = j
        finally:
            os.close(fd)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Random access in shard order, for inspection (e.g. the augmentation preview); training iterates."""
        position = int(np.searchsorted(np.cumsum(self.counts), i, side="right"))
        first = i - sum(self.counts[:position])
        return self._decode(next(self._read_segment(position, first, first + 1)))


class ShardWriter:
    """Write one split of a shard set: call ``add`` per sample in the final (already shuffled) order, then ``close``."""

    def __init__(self, root: str | os.PathLike, split: str, label_columns: list[str], max_bytes: int = 2**30):  # noqa: D107
        self.root, self.split, self.label_columns, self.max_bytes = Path(root), split, label_columns, max_bytes
        (self.root / split).mkdir(parents=True, exist_ok=True)
        self.shards: list[dict] = []
        self._start()

    def _start(self) -> None:
        self.name = f"{self.split}/{len(self.shards):06d}"
        self.partial = self.root / f"{self.name}.tar.partial"
        self.tar = open(self.partial, "wb")
        self.rows: dict[str, list] = {c: [] for c in (*INDEX_COLUMNS, *self.label_columns)}
        self.position, self.digest = 0, hashlib.sha256()

    def _write(self, data: bytes) -> None:
        self.tar.write(data)
        self.digest.update(data)
        self.position += len(data)

    def add(self, key: str, data: bytes, labels: dict[str, str]) -> None:
        """Append one sample; ``labels`` maps every label column to its value."""
        info = tarfile.TarInfo(f"{key}.jpg")
        info.size = len(data)
        self._write(info.tobuf(format=tarfile.USTAR_FORMAT))
        self.rows["key"].append(key)
        self.rows["offset"].append(self.position)
        self.rows["size"].append(len(data))
        self.rows["crc32"].append(zlib.crc32(data))
        for column in self.label_columns:
            self.rows[column].append(str(labels[column]))
        self._write(data + b"\0" * (-len(data) % HEADER))
        if self.position >= self.max_bytes:
            self._finish()
            self._start()

    def _finish(self) -> None:
        pa, pq = _parquet()
        self._write(b"\0" * (2 * HEADER))
        self.tar.close()
        table = pa.table(self.rows)
        partial_index = self.root / f"{self.name}.parquet.partial"
        pq.write_table(table, partial_index)
        os.replace(partial_index, self.root / f"{self.name}.parquet")
        os.replace(self.partial, self.root / f"{self.name}.tar")
        self.shards.append({"name": self.name, "n_samples": table.num_rows, "tar_bytes": self.position, "sha256": self.digest.hexdigest()})

    def close(self) -> dict:
        """Finish the last shard and return this split's manifest entry."""
        if self.rows["key"]:
            self._finish()
        else:
            self.tar.close()
            self.partial.unlink()
        return {"n_samples": sum(s["n_samples"] for s in self.shards), "shards": self.shards}


def write_manifest(root: str | os.PathLike, splits: dict[str, dict], label_columns: list[str], source: dict | None = None) -> None:
    """Publish a complete shard set atomically: the manifest appears only after every shard exists."""
    path = os.path.join(root, "manifest.json")
    with open(f"{path}.partial", "w") as f:
        json.dump(
            {"format": FORMAT, "version": VERSION, "source": source or {}, "label_columns": label_columns, "splits": splits}, f, indent=1
        )
    os.replace(f"{path}.partial", path)


def verify(root: str | os.PathLike, full: bool = False) -> int:
    """Check a shard set and return its sample count; ``full`` also reads every sample (header, CRC) and SHA-256."""
    manifest = load_manifest(root)
    total = 0
    for split, entry in manifest["splits"].items():
        dataset = ShardDataset(root, split, lambda data: torch.empty(0), {"0": _IdentityClasses()}, 1, train=False)
        for position, shard in enumerate(entry["shards"]):
            fd, path = dataset._open(position)
            os.close(fd)
            if full:
                if sum(1 for _ in dataset._read_segment(position, 0, shard["n_samples"])) != shard["n_samples"]:
                    raise ShardError(f"{path}: wrong sample count.")
                digest = hashlib.sha256()
                with open(path, "rb") as f:
                    while chunk := f.read(2**24):
                        digest.update(chunk)
                if digest.hexdigest() != shard["sha256"]:
                    raise ShardError(f"{path}: SHA-256 mismatch.")
            else:
                dataset._index(position)
            total += shard["n_samples"]
    return total


class _IdentityClasses(dict):
    """A class map accepting every label, for verification without a class specification."""

    def __missing__(self, key):
        return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check a shard set.")
    parser.add_argument("command", choices=["verify"])
    parser.add_argument("root")
    parser.add_argument("--full", action="store_true", help="Read every sample and hash every tar.")
    args = parser.parse_args()
    print(f"{args.root}: {verify(args.root, full=args.full):,} samples OK")
