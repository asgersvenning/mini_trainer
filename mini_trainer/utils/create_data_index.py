import sys
from argparse import ArgumentParser
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

from mini_trainer.data import (
    collect_samples_from_source,
    create_metadata,
)
from mini_trainer.integrations import resolve_taxonomical_classes


def parse_verbosity(verbosity: int | str) -> int:
    """Parse verbosity level into an integer (0=minimal, 1=info, 2=debug)."""
    if isinstance(verbosity, int):
        return max(0, min(2, verbosity))
    v = str(verbosity).strip().lower()
    if v in ("0", "minimal", "error", "errors", "quiet", "q"):
        return 0
    if v in ("1", "info", "normal", "default", "v"):
        return 1
    if v in ("2", "debug", "verbose", "vv"):
        return 2
    try:
        return max(0, min(2, int(v)))
    except ValueError:
        return 1


def create_data_index_file(
    sources: list[str | Path | dict] | str | Path | dict,
    output: str | Path | None = None,
    train_proportion: float = 0.8,
    test_proportion: float = 0.2,
    min_train: int = 0,
    min_test: int = 1,
    no_gbif: bool = False,
    levels: str | int | Iterable[str | int] | None = None,
    verbosity: int | str = "info",
    seed: int | None = 42,
    relative_paths: bool = True,
) -> dict[str, list]:
    """Generate a static data index with consistent train-test splitting."""
    verb = parse_verbosity(verbosity)
    src_list = sources if isinstance(sources, list) else [sources]

    if verb >= 1:
        print(f"[INFO] Collecting samples from {len(src_list)} source(s)...")

    if verb >= 2:
        raw_samples: list[tuple[str, str, str | None]] = []
        for src in src_list:
            items = collect_samples_from_source(src)
            raw_samples.extend(items)
            print(f"[DEBUG] Source '{src}' contributed {len(items)} samples.", file=sys.stderr)
    else:
        raw_samples = collect_samples_from_source(src_list)

    if not raw_samples:
        raise ValueError(f"No samples could be extracted from sources: {sources}")

    unique_raw_classes = sorted(set(s[1] for s in raw_samples))
    if verb >= 1:
        print(f"[INFO] Loaded {len(raw_samples)} samples across {len(unique_raw_classes)} raw classes.")

    labels_map = None
    if no_gbif:
        if verb >= 1:
            print("[INFO] GBIF taxonomy resolution disabled (--no-gbif). Using raw dataset class labels.")
    else:
        if verb >= 1:
            print(f"[INFO] Resolving {len(unique_raw_classes)} classes via GBIF taxonomy...")
        labels_map, collapsed_classes = resolve_taxonomical_classes(unique_raw_classes, levels=levels)
        if collapsed_classes:
            total_collapsed = sum(len(raws) for raws in collapsed_classes.values())
            if verb >= 1:
                print(
                    f"[INFO] GBIF taxonomy resolution collapsed {total_collapsed} raw classes "
                    f"into {len(collapsed_classes)} canonical classes:"
                )
                for resolved, raws in sorted(collapsed_classes.items(), key=lambda x: str(x[0])):
                    print(f"  - {resolved!r} <- [{', '.join(repr(r) for r in raws)}]")
        elif verb >= 1:
            print("[INFO] No class collapses detected during GBIF resolution.")

    min_freqs = {"train": int(min_train), "test": int(min_test)}
    is_presplit = len(raw_samples) > 0 and all(len(s) >= 3 and s[2] is not None for s in raw_samples)
    if is_presplit and verb >= 1:
        detected_splits = sorted(set(s[2] for s in raw_samples if len(s) >= 3 and s[2] is not None))
        print(f"[INFO] Detected pre-split dataset ({', '.join(detected_splits)}). Preserving existing split assignments.")
    elif not is_presplit and verb >= 1:
        print(
            f"[INFO] Splitting dataset (train={train_proportion:.1%}, test={test_proportion:.1%}, "
            f"min_train={min_train}, min_test={min_test})..."
        )

    # Delegate to create_metadata
    metadata = create_metadata(
        directory=raw_samples,
        labels=labels_map,
        train_proportion=train_proportion,
        test_proportion=test_proportion,
        val_proportion=0.0,
        min_freqs=min_freqs,
        output=output,
        relative_paths=relative_paths,
        seed=seed,
    )

    if verb >= 1:
        counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for p, s, lbl in zip(metadata["path"], metadata["split"], metadata["label"]):
            counts[s][lbl] += 1

        total_samples = len(metadata["path"])
        print("[INFO] Split summary:")
        for s in ("train", "test"):
            total_s = sum(counts[s].values())
            n_classes_s = len(counts[s])
            pct_s = (total_s / total_samples) if total_samples > 0 else 0.0
            print(f"  - {s:>5}: {total_s:>6} samples ({pct_s:>5.1%}) across {n_classes_s:>5} classes")

        if is_presplit:
            target_map = {"train": train_proportion, "test": test_proportion}
            diffs = []
            for s in ("train", "test"):
                actual_pct = (sum(counts[s].values()) / total_samples) if total_samples > 0 else 0.0
                diffs.append(f"{s}: actual={actual_pct:.1%} vs requested target={target_map[s]:.1%}")
            print(f"[INFO] Pre-split proportion alignment: {'; '.join(diffs)}")

        all_classes = sorted(set(metadata["label"]))
        violations = {s: 0 for s in ("train", "test")}
        for cls in all_classes:
            for s in ("train", "test"):
                if counts[s].get(cls, 0) < min_freqs[s]:
                    violations[s] += 1

        total_viol = sum(1 for cls in all_classes if any(counts[s].get(cls, 0) < min_freqs[s] for s in ("train", "test")))
        if total_viol > 0:
            print(
                f"[INFO] Minimum frequency violations: {total_viol} of {len(all_classes)} "
                f"classes violated minimum split-class requirements:"
            )
            for s in ("train", "test"):
                if violations[s] > 0:
                    print(f"  - {s:>5} (< {min_freqs[s]}): {violations[s]} classes")
        else:
            print("[INFO] Minimum frequency requirements: all classes satisfied requested minimum frequencies.")

        if output is not None:
            print(f"[INFO] Saved data index to '{output}'.")

    return metadata


def main(
    sources: list[str] | str,
    output: str | None = None,
    train_proportion: float = 0.8,
    test_proportion: float = 0.2,
    min_train: int = 0,
    min_test: int = 1,
    no_gbif: bool = False,
    levels: str | int | list[str | int] | None = None,
    verbosity: int | str = "info",
    seed: int | None = 42,
    relative_paths: bool = True,
):
    return create_data_index_file(
        sources=sources,
        output=output,
        train_proportion=train_proportion,
        test_proportion=test_proportion,
        min_train=min_train,
        min_test=min_test,
        no_gbif=no_gbif,
        levels=levels,
        verbosity=verbosity,
        seed=seed,
        relative_paths=relative_paths,
    )


def cli():
    parser = ArgumentParser(
        prog="mt_create_data_index",
        description="Generate a static data index with consistent train-test splitting and GBIF taxonomy resolution.",
    )
    parser.add_argument(
        "-i",
        "--input",
        dest="sources",
        nargs="+",
        type=str,
        required=True,
        help="Input dataset source(s): directory with class subdirectories, Parquet file, or JSON data index.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output file path for data_index.json (or directory).",
    )
    parser.add_argument(
        "--train-proportion",
        "--train_proportion",
        dest="train_proportion",
        type=float,
        default=0.8,
        help="Target proportion of samples for the train split (default: 0.8).",
    )
    parser.add_argument(
        "--test-proportion",
        "--test_proportion",
        dest="test_proportion",
        type=float,
        default=0.2,
        help="Target proportion of samples for the test split (default: 0.2).",
    )
    parser.add_argument(
        "--min-train",
        "--min_train",
        dest="min_train",
        type=int,
        default=0,
        help="Attempted minimum sample frequency per class in the train split (default: 0).",
    )
    parser.add_argument(
        "--min-test",
        "--min_test",
        dest="min_test",
        type=int,
        default=1,
        help="Attempted minimum sample frequency per class in the test split (default: 1).",
    )
    parser.add_argument(
        "--no-gbif",
        "--no_gbif",
        action="store_true",
        dest="no_gbif",
        help="Disable GBIF taxonomy resolution. Use raw dataset class labels directly.",
    )
    parser.add_argument(
        "--levels",
        nargs="*",
        default=None,
        help="Specific taxonomy levels to use (e.g. species, genus). Default: automatic via select_levels.",
    )
    parser.add_argument(
        "-v",
        "--verbosity",
        default="info",
        help="Logging verbosity level: 0/minimal, 1/info (default), 2/debug.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sample splitting (default: 42).",
    )
    parser.add_argument(
        "--absolute",
        action="store_false",
        dest="relative_paths",
        help="Store absolute image paths instead of paths relative to the data index.",
    )

    args = vars(parser.parse_args())
    if len(args["sources"]) == 1:
        args["sources"] = args["sources"][0]
    if args.get("levels") is not None and len(args["levels"]) == 1:
        args["levels"] = args["levels"][0]
    return args


def run():
    main(**cli())


if __name__ == "__main__":
    run()
