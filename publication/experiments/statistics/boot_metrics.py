# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "mini_metrics @ git+https://github.com/GuillaumeMougeot/mini_metrics.git",
#   "tqdm"
# ]
# ///

"""Repeat seeded mini_metrics calibration/evaluation; no wrapper-level resampling."""

import random
from argparse import ArgumentParser
from csv import DictWriter

import mini_metrics
import mini_metrics.metrics
from mini_metrics.data import MetricDF
from mini_metrics.metrics import evaluate_file
from tqdm.auto import tqdm


def proc_one(
    src: MetricDF, seed: int, combinations: str | None = None, label_filter: str | list[str] | None = None, subsample: int | None = None
):
    metrics = evaluate_file(
        source=src,
        combinations=combinations,
        optimal=True,
        threshold=None,
        known_only=False,
        label_filter=label_filter,
        subsample=subsample,
        per_class=False,
        pattern=None,
        simple=True,
        hierarchical=False,
        seed=seed,
        opt_crit=mini_metrics.metrics.MacroF1,
        use_quantiles=True,
        eps=5e-2,
        verbose=0,
    )
    levels = len(next(iter(metrics.values())))
    retval = {"id": [seed] * levels, "level": list(range(levels))}
    for name, lvl_val in metrics.items():
        retval[name] = [None] * levels
        for lvl, val in sorted(lvl_val.items(), key=lambda lv: lv[0]):
            retval[name][lvl] = val
        for lvl, val in enumerate(retval[name]):
            if val is None:
                raise RuntimeError(f"No value found for metric={name} at level={lvl}!")
    return retval


def combine[V](*dicts: dict[str, list[V]]):
    out: dict[str, list[V]] = {}
    for d in dicts:
        for k, v in d.items():
            if k not in out:
                out[k] = []
            out[k].extend(v)
    return out


def write_dict_to_csv(filename: str, data: dict[str, list]) -> None:
    fields = data.keys()
    with open(filename, "w", newline="") as f:
        writer = DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(dict(zip(fields, row)) for row in sorted(zip(*data.values())))


def main(file: str, dst: str, n: int, seed: int | None = None, combinations: str | None = None):
    rng = random.Random(seed)
    seeds = [rng.getrandbits(64) for _ in range(n)]
    data = MetricDF.from_source(file)
    cfg = {"combinations": combinations}
    results = combine(*(proc_one(data, seed, **cfg) for seed in tqdm(seeds, desc="Evaluating calibration splits", unit="it")))
    write_dict_to_csv(dst, results)


def cli():
    parser = ArgumentParser("boot-metrics", description="Repeat mini_metrics threshold calibration and report metrics by seed and level.")
    parser.add_argument(
        "-i",
        "-I",
        "--input",
        "--file",
        type=str,
        required=True,
        dest="file",
        help="Input file with predictions and labels following the `mini-metric.MetricDF` specification.",
    )
    parser.add_argument(
        "-o",
        "-O",
        "--output",
        "--dst",
        type=str,
        required=True,
        dest="dst",
        help="Output CSV of metrics by evaluation seed and taxonomic level.",
    )
    parser.add_argument(
        "-n",
        "--n",
        type=int,
        default=100,
        required=False,
        dest="n",
        help="Number of seeded evaluations; each recalibrates thresholds (default: 100).",
    )
    parser.add_argument(
        "-s",
        "--seed",
        "--global_seed",
        "--global-seed",
        type=int,
        default=None,
        required=False,
        help="Global seed used to generate reproducible per-evaluation seeds.",
    )
    parser.add_argument(
        "-C",
        "--combinations",
        type=str,
        default=None,
        required=False,
        help="Optional CSV mapping leaf classes to higher taxonomy ranks.",
    )
    return vars(parser.parse_args())


if __name__ == "__main__":
    main(**cli())
