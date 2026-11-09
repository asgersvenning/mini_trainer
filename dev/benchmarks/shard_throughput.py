"""Build a shard subset from loose images, and measure sustained shard loading through the real loader.

    python dev/benchmarks/shard_throughput.py build --parquet DATA.parquet --images IMAGES --out ROOT --count N
    python dev/benchmarks/shard_throughput.py measure --root ROOT --ranks 8 --workers 44 --batch-size 96 --read-size 384 --epochs 2

``measure`` starts one process per rank (real DDP setup on CPU, gloo), decodes and crops every sample, trains nothing,
and reports images/second per epoch. The first epoch in a fresh job is the cold read and later ones are warm.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from multiprocessing.pool import ThreadPool

import torch

from mini_trainer.data import ShardWriter, get_shard_dataloaders, write_manifest
from mini_trainer.data.shards import load_manifest
from mini_trainer.integrations.parquet import KCOLUMNS
from mini_trainer.utils._concurrent.distributed import get_rank, init_distributed, reduce_across_processes


def build(a):
    """Shuffle ``count`` rows of a gbifxdl parquet (set >= 1 as train, set 0 as val) into a shard set, bytes unchanged."""
    import pyarrow.parquet as pq

    rows = pq.read_table(a.parquet, columns=["filename", "set", *KCOLUMNS]).to_pandas()
    rows = rows.sample(n=min(a.count, len(rows)), random_state=a.seed).reset_index(drop=True)
    writers = {
        "train": ShardWriter(a.out, "train", list(KCOLUMNS), a.shard_bytes),
        "val": ShardWriter(a.out, "val", list(KCOLUMNS), a.shard_bytes),
    }

    def read(row):
        try:
            with open(os.path.join(a.images, str(row.speciesKey), row.filename), "rb") as f:
                return f.read()
        except FileNotFoundError:
            return None

    missing, block = 0, 4096
    with ThreadPool(a.read_threads) as pool:
        for start in range(0, len(rows), block):
            part = list(rows.iloc[start : start + block].itertuples(index=False))
            for row, data in zip(part, pool.map(read, part)):
                if data is None:
                    missing += 1
                    continue
                key = os.path.splitext(row.filename)[0]
                writers["val" if str(row.set) == "0" else "train"].add(key, data, {k: getattr(row, k) for k in KCOLUMNS})
    print(f"{len(rows) - missing:,} images written, {missing:,} source files missing")
    write_manifest(
        a.out, {s: w.close() for s, w in writers.items()}, list(KCOLUMNS), {"parquet": a.parquet, "count": a.count, "seed": a.seed}
    )
    manifest = load_manifest(a.out)
    print(json.dumps({s: {"samples": v["n_samples"], "shards": len(v["shards"])} for s, v in manifest["splits"].items()}))


class _AnyClass(dict):
    def __missing__(self, key):
        return 0


def rank_main(a):
    init_distributed(device="cpu")
    spec = {"0": _AnyClass(a=0)}
    train = get_shard_dataloaders(
        a.root, spec, a.read_size, modes=("train",), batch_size=a.batch_size, num_workers=a.workers, multilabel=True,
        resize_method="crop", read_size=a.read_size, shard_shuffle_buffer=a.shuffle_buffer,
    )[0]  # fmt: skip
    results = []
    for epoch in range(a.epochs):
        train.dataset.set_epoch(epoch)
        torch.distributed.barrier()
        start, images = time.perf_counter(), 0
        for batch, _ in train:
            images += len(batch)
        seconds = time.perf_counter() - start
        total = int(reduce_across_processes(images))
        torch.distributed.barrier()
        slowest = float(reduce_across_processes(seconds))  # sum over ranks; divided below
        if get_rank() == 0:
            mean = slowest / torch.distributed.get_world_size()
            results.append(
                {
                    "epoch": epoch,
                    "kind": "cold" if epoch == 0 else "warm",
                    "images": total,
                    "mean_seconds_per_rank": mean,
                    "images_per_second": total / mean,
                }
            )
            print(json.dumps(results[-1]), flush=True)
    if get_rank() == 0 and a.output:
        with open(a.output, "w") as f:
            json.dump({"arguments": vars(a), "results": results, "disk_note": "first epoch in a fresh job is the cold read"}, f, indent=1)


def measure(a):
    env = {**os.environ, "WORLD_SIZE": str(a.ranks), "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(a.port), "CUDA_VISIBLE_DEVICES": ""}
    arguments = [
        sys.executable,
        __file__,
        "rank",
        *sum(
            ([f"--{k.replace('_', '-')}", str(v)] for k, v in vars(a).items() if k not in ("command", "ranks", "port") and v is not None),
            [],
        ),
    ]
    processes = [subprocess.Popen(arguments, env={**env, "RANK": str(r), "LOCAL_RANK": str(r)}) for r in range(a.ranks)]
    codes = [p.wait() for p in processes]
    if any(codes):
        raise SystemExit(f"Rank processes failed: {codes}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--parquet", required=True), b.add_argument("--images", required=True), b.add_argument("--out", required=True)
    (
        b.add_argument("--count", type=int, required=True),
        b.add_argument("--seed", type=int, default=0),
        b.add_argument("--shard-bytes", type=int, default=2**30),
        b.add_argument("--read-threads", type=int, default=32),
    )
    for name in ("measure", "rank"):
        m = sub.add_parser(name)
        (
            m.add_argument("--root", required=True),
            m.add_argument("--workers", type=int, default=8),
            m.add_argument("--batch-size", type=int, default=96),
        )
        (
            m.add_argument("--read-size", type=int, default=384),
            m.add_argument("--epochs", type=int, default=2),
            m.add_argument("--shuffle-buffer", type=int, default=2048),
        )
        m.add_argument("--output")
        if name == "measure":
            m.add_argument("--ranks", type=int, default=1), m.add_argument("--port", type=int, default=29541)
    args = parser.parse_args()
    {"build": build, "measure": measure, "rank": rank_main}[args.command](args)
