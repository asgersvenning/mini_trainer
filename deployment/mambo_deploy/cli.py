"""Small local prediction CLI; no dataset or training-framework dependency."""

import argparse
import csv
import json
import tempfile
from contextlib import closing
from pathlib import Path

import numpy as np

from .augmentation import DEFAULT_TTA, PROFILES
from .predictor import Predictor


def run(default_backend="onnx", default_device="cpu"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--input", nargs="+", required=True)
    parser.add_argument("--bundle")
    parser.add_argument("--backend", choices=["torch", "onnx"], default=default_backend)
    parser.add_argument("--device", default=default_device)
    parser.add_argument("-M", "--model")
    parser.add_argument("-w", "--weights")
    parser.add_argument("--class-list")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--precision", choices=["auto", "fp32", "fp16", "bf16", "tf32"], default="auto")
    parser.add_argument(
        "--tta",
        nargs="?",
        const=DEFAULT_TTA,
        choices=PROFILES,
        default="none",
        help=f"Enable TTA (default recipe: {DEFAULT_TTA}), or choose a recipe",
    )
    parser.add_argument("--preprocess-workers", type=int, help="Preparation threads; defaults to --threads")
    parser.add_argument("--topk", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0)
    parser.add_argument("--embeddings", action="store_true")
    parser.add_argument("-o", "--output", type=Path, default=Path("."))
    parser.add_argument("--name", default="results")
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1:
        parser.error("threshold must be in [0,1]")
    paths = []
    for value in args.input:
        path = Path(value)
        paths.extend(
            sorted(
                p
                for p in path.rglob("*")
                if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
            )
            if path.is_dir()
            else [path]
        )
    destination = args.output / args.name
    if destination.exists():
        parser.error(f"Output already exists: {destination}; choose a new --name")
    if not paths:
        parser.error("No images found")
    predictor = Predictor(
        args.bundle,
        backend=args.backend,
        device=args.device,
        model=args.model,
        weights=args.weights,
        class_list=args.class_list,
        batch_size=args.batch_size,
        threads=args.threads,
        precision=args.precision,
        tta=args.tta,
        preprocess_workers=args.preprocess_workers,
    )
    columns = [
        "instance_id",
        "filename",
        "level",
        "label",
        "prediction",
        "confidence",
        "threshold",
        "known_label",
        "prediction_made",
        "correct",
    ]
    full_labels = predictor.bundle.classes["labels"]
    species_index = {label: i for i, label in enumerate(full_labels[0])}
    parents = predictor.bundle.classes["parents"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Publish only a complete request. Failures leave no plausible final results.
    with tempfile.TemporaryDirectory(prefix=".mambo-results-", dir=destination.parent) as temporary:
        output = Path(temporary) / "results"
        output.mkdir()
        with (
            (output / "mini_metric.csv").open("w", newline="") as stream,
            (output / "predictions.json").open("w") as predictions,
            closing(predictor.predict_stream(paths, topk=args.topk, embeddings=args.embeddings)) as batches,
        ):
            writer = csv.writer(stream)
            writer.writerow(columns)
            offset, vectors = 0, None
            try:
                for batch in batches:
                    result, embedding = batch if args.embeddings else (batch, None)
                    if offset == 0:
                        predictions.write('{"metadata": ' + json.dumps(result.metadata))
                        predictions.write(', "config": ' + json.dumps({"topk": result.topk, "cls2idx": result.cls2idx}))
                        predictions.write(', "results": [')
                        if embedding is not None:
                            vectors = np.lib.format.open_memmap(
                                output / "embeddings.npy", mode="w+", dtype=np.float32, shape=(len(paths), embedding.shape[1])
                            )
                    for i, record in enumerate(result.to_dict()):
                        if offset + i:
                            predictions.write(",")
                        predictions.write(json.dumps(record))
                        path = paths[offset + i]
                        truth = [path.parent.name, "", ""]
                        if truth[0] in species_index:
                            genus = parents[0][species_index[truth[0]]]
                            truth[1:] = [full_labels[1][genus], full_labels[2][parents[1][genus]]]
                        for rank in range(3):
                            label = result.labels[i][0][rank]
                            confidence = float(result.confidence[i, 0, rank])
                            known = truth[rank] in result.cls2idx[str(rank)]
                            made = confidence >= args.threshold
                            writer.writerow(
                                [
                                    offset + i,
                                    str(path),
                                    rank,
                                    truth[rank],
                                    label,
                                    confidence,
                                    args.threshold,
                                    int(known),
                                    int(made),
                                    (1 if label == truth[rank] else -1) if made else 0,
                                ]
                            )
                    if vectors is not None:
                        vectors[offset : offset + len(result)] = embedding
                    offset += len(result)
                if offset != len(paths):
                    raise RuntimeError(f"Incomplete prediction: {offset}/{len(paths)} images")
                predictions.write("]}")
            finally:
                if vectors is not None:
                    vectors.flush()
                    # Release the mapping before directory publication on Windows.
                    del vectors
        output.rename(destination)
    print(destination)
