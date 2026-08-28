import json
import os
import random
import sys
from argparse import ArgumentParser
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from mini_trainer.data import (
    collect_samples_from_source,
    partition_class_samples,
)
from mini_trainer.integrations.gbif import resolve_name_or_id


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


def resolve_classes_gbif(
    raw_labels: Sequence[str],
    verbosity: int = 1,
) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Resolve raw dataset class identifiers to canonical GBIF taxonomy and detect class collapse.

    Returns:
        raw_to_resolved: mapping from raw label -> canonical resolved label.
        collapsed_classes: mapping from resolved label -> list of raw labels (for groups where len > 1).
    """
    raw_to_resolved: dict[str, str] = {}
    unique_raw = sorted(set(raw_labels))

    try:
        taxs = resolve_name_or_id(unique_raw)
        for raw, tax in zip(unique_raw, taxs):
            if "species" in tax:
                resolved = str(tax["species"][1])
            elif len(tax) > 0:
                resolved = str(next(iter(tax.values()))[1])
            else:
                resolved = str(raw)
            raw_to_resolved[raw] = resolved
    except Exception as e:
        if verbosity >= 2:
            print(f"[DEBUG] Batch GBIF resolution encountered exception ({e}); resolving individually...", file=sys.stderr)
        for raw in unique_raw:
            try:
                tax = resolve_name_or_id(raw)
                if isinstance(tax, list) and len(tax) > 0:
                    tax = tax[0]
                if isinstance(tax, dict) and "species" in tax:
                    resolved = str(tax["species"][1])
                elif isinstance(tax, dict) and len(tax) > 0:
                    resolved = str(next(iter(tax.values()))[1])
                else:
                    resolved = str(raw)
            except Exception as ex:
                if verbosity >= 2:
                    print(f"[DEBUG] Could not resolve '{raw}' via GBIF ({ex}); retaining raw label.", file=sys.stderr)
                resolved = str(raw)
            raw_to_resolved[raw] = resolved

    resolved_to_raw: dict[str, list[str]] = defaultdict(list)
    for raw, resolved in raw_to_resolved.items():
        resolved_to_raw[resolved].append(raw)

    collapsed_classes = {resolved: raws for resolved, raws in resolved_to_raw.items() if len(raws) > 1}
    return raw_to_resolved, collapsed_classes


def create_data_index_file(
    sources: list[str | Path | dict] | str | Path | dict,
    output: str | Path | None = None,
    train_proportion: float = 0.8,
    test_proportion: float = 0.2,
    min_train: int = 0,
    min_test: int = 1,
    no_gbif: bool = False,
    verbosity: int | str = "info",
    seed: int | None = 42,
    relative_paths: bool = True,
) -> dict[str, list]:
    """Generate a static data index with consistent train-test splitting.

    Args:
        sources: Dataset source or list of sources (directories, Parquet, JSON data index).
        output: Destination file path for data_index.json (or directory). If None, returns metadata dict without writing.
        train_proportion: Target proportion for the training split (default: 0.8).
        test_proportion: Target proportion for the test split (default: 0.2).
        min_train: Attempted minimum sample frequency per class in the train split (default: 0).
        min_test: Attempted minimum sample frequency per class in the test split (default: 1).
        no_gbif: If True, disables GBIF taxonomic resolution and uses raw dataset class labels.
        verbosity: Logging verbosity level (0/minimal, 1/info, 2/debug).
        seed: Random seed for reproducible sample shuffling and splitting.
        relative_paths: Whether to store paths relative to the output index directory (default: True).

    Returns:
        metadata: Dictionary with keys 'path', 'class', 'split', 'label'.
    """
    verb = parse_verbosity(verbosity)
    rng = random.Random(seed)

    if not isinstance(sources, list):
        sources = [sources]

    if verb >= 1:
        print(f"[INFO] Collecting samples from {len(sources)} source(s)...")

    raw_samples: list[tuple[str, str]] = []
    for src in sources:
        items = collect_samples_from_source(src)
        raw_samples.extend(items)
        if verb >= 2:
            print(f"[DEBUG] Source '{src}' contributed {len(items)} samples.", file=sys.stderr)

    if not raw_samples:
        raise ValueError(f"No samples could be extracted from sources: {sources}")

    total_samples = len(raw_samples)
    unique_raw_classes = sorted(set(lbl for _, lbl in raw_samples))
    if verb >= 1:
        print(f"[INFO] Loaded {total_samples} samples across {len(unique_raw_classes)} raw classes.")

    # GBIF taxonomic resolution
    collapsed_classes: dict[str, list[str]] = {}
    if no_gbif:
        if verb >= 1:
            print("[INFO] GBIF taxonomy resolution disabled (--no-gbif). Using raw dataset class labels.")
        raw_to_resolved = {c: c for c in unique_raw_classes}
    else:
        if verb >= 1:
            print(f"[INFO] Resolving {len(unique_raw_classes)} classes via GBIF taxonomy...")
        raw_to_resolved, collapsed_classes = resolve_classes_gbif(unique_raw_classes, verbosity=verb)

        if collapsed_classes:
            total_raw_collapsed = sum(len(raws) for raws in collapsed_classes.values())
            if verb >= 1:
                print(
                    f"[INFO] GBIF taxonomy resolution collapsed {total_raw_collapsed} raw classes "
                    f"into {len(collapsed_classes)} canonical classes:"
                )
                for resolved, raws in sorted(collapsed_classes.items()):
                    print(f"  - '{resolved}' <- [{', '.join(repr(r) for r in raws)}]")
        elif verb >= 1:
            print("[INFO] No class collapses detected during GBIF resolution.")

    # Group sample image paths by resolved class
    samples_by_class: dict[str, list[str]] = defaultdict(list)
    for img_path, raw_lbl in raw_samples:
        resolved_lbl = raw_to_resolved.get(raw_lbl, raw_lbl)
        samples_by_class[resolved_lbl].append(img_path)

    all_resolved_classes = sorted(samples_by_class.keys())
    cls2idx = {cls_name: idx for idx, cls_name in enumerate(all_resolved_classes)}

    proportions = {"train": float(train_proportion), "test": float(test_proportion)}
    min_freqs = {"train": int(min_train), "test": int(min_test)}

    if verb >= 1:
        print(
            f"[INFO] Splitting dataset (train={proportions['train']:.1%}, test={proportions['test']:.1%}, "
            f"min_train={min_freqs['train']}, min_test={min_freqs['test']})..."
        )

    split_paths: dict[str, list[str]] = defaultdict(list)
    split_labels: dict[str, list[str]] = defaultdict(list)
    split_classes: dict[str, list[int]] = defaultdict(list)
    split_class_occurrences: dict[str, set[str]] = defaultdict(set)

    total_violations_per_split: dict[str, int] = defaultdict(int)
    classes_with_violations: list[tuple[str, dict[str, int], dict[str, int]]] = []

    for cls_name in all_resolved_classes:
        cls_items = samples_by_class[cls_name]
        class_splits, class_violations = partition_class_samples(
            samples=cls_items,
            proportions=proportions,
            min_freqs=min_freqs,
            rng=rng,
        )

        has_violation = False
        actual_counts = {}
        for s, s_samples in class_splits.items():
            actual_counts[s] = len(s_samples)
            if class_violations[s] > 0:
                has_violation = True
                total_violations_per_split[s] += 1

            if s_samples:
                split_class_occurrences[s].add(cls_name)
                split_paths[s].extend(s_samples)
                split_labels[s].extend([cls_name] * len(s_samples))
                split_classes[s].extend([cls2idx[cls_name]] * len(s_samples))

        if has_violation:
            classes_with_violations.append((cls_name, actual_counts, class_violations))

    # Form final metadata lists
    output_path = None
    if output is not None:
        out_p = Path(output).expanduser().resolve()
        if out_p.is_dir() or (not out_p.exists() and out_p.suffix == ""):
            out_p.mkdir(parents=True, exist_ok=True)
            output_path = out_p / "data_index.json"
        else:
            out_p.parent.mkdir(parents=True, exist_ok=True)
            output_path = out_p

    base_dir = output_path.parent if output_path is not None else Path(os.getcwd())

    final_paths: list[str] = []
    final_classes: list[int] = []
    final_splits: list[str] = []
    final_labels: list[str] = []

    for s in ("train", "test"):
        for p, c, lbl in zip(split_paths[s], split_classes[s], split_labels[s]):
            formatted_p = os.path.relpath(os.path.abspath(p), base_dir) if relative_paths else os.path.abspath(p)
            final_paths.append(formatted_p)
            final_classes.append(c)
            final_splits.append(s)
            final_labels.append(lbl)

    metadata = {
        "path": final_paths,
        "class": final_classes,
        "split": final_splits,
        "label": final_labels,
    }

    # Summary reporting
    if verb >= 1:
        print("[INFO] Split summary:")
        for s in ("train", "test"):
            print(f"  - {s:>5}: {len(split_paths[s]):>6} samples across {len(split_class_occurrences[s]):>5} classes")

        if classes_with_violations:
            print(
                f"[INFO] Minimum frequency violations: {len(classes_with_violations)} of {len(all_resolved_classes)} "
                f"classes violated minimum split-class requirements:"
            )
            for s in ("train", "test"):
                if total_violations_per_split[s] > 0:
                    print(f"  - {s:>5} (< {min_freqs[s]}): {total_violations_per_split[s]} classes")

            if verb >= 2:
                print("[DEBUG] Detailed class violations:", file=sys.stderr)
                for cls_name, counts, viols in classes_with_violations:
                    counts_str = ", ".join(f"{s}={counts[s]} (min={min_freqs[s]})" for s in ("train", "test"))
                    print(f"  - Class '{cls_name}': {counts_str}", file=sys.stderr)
        else:
            print("[INFO] Minimum frequency requirements: all classes satisfied requested minimum frequencies.")

    if output_path is not None:
        with open(output_path, "w", encoding="utf8") as f:
            json.dump(metadata, f, indent=2)
        if verb >= 1:
            print(f"[INFO] Saved data index to '{output_path}'.")

    return metadata


def main(
    sources: list[str] | str,
    output: str | None = None,
    train_proportion: float = 0.8,
    test_proportion: float = 0.2,
    min_train: int = 0,
    min_test: int = 1,
    no_gbif: bool = False,
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
    return args


def run():
    main(**cli())


if __name__ == "__main__":
    run()
