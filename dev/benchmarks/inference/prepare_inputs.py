"""Prepare calibration or held-out image batches from explicit export/dataset contracts."""

import hashlib
import importlib
import inspect
import json
import random
from argparse import ArgumentParser
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from .dataset_inference import inference_manifest
from .onnx_calibration import calibration_manifest
from .onnx_inference import file_hash


@contextmanager
def seeded(seed):
    """Restore caller RNGs; factories and transforms get explicit independent seeds."""
    import torch

    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[]):
            random.seed(seed)
            np.random.seed(seed % 2**32)
            torch.manual_seed(seed)
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def ordered_classes(mapping):
    if not isinstance(mapping, dict) or not mapping or any(not isinstance(k, str) or not k for k in mapping):
        raise ValueError("Require nonempty class-name mappings")
    if all(type(v) is int for v in mapping.values()):
        if sorted(mapping.values()) != list(range(len(mapping))):
            raise ValueError("Class indices must be unique and contiguous from zero")
        return [sorted(mapping, key=mapping.get)]
    if set(mapping) != {str(i) for i in range(len(mapping))}:
        raise ValueError("Hierarchical level indices must be contiguous from zero")
    result = []
    for i in range(len(mapping)):
        level = ordered_classes(mapping[str(i)])
        if len(level) != 1:
            raise ValueError("Expected one class mapping per level")
        result.extend(level)
    return result


def select_records(dataset, split, count, seed):
    if dataset.get("schema_version") != 1 or split not in ("train", "val", "test"):
        raise ValueError("Require a version-1 dataset and train/val/test split")
    seen, hashes, selected = set(), {}, []
    for record in dataset["records"]:
        path, digest, source_split = record["path"], record["sha256"], record["split"]
        if not isinstance(path, str) or not path or path in seen:
            raise ValueError("Dataset paths must be nonempty and unique")
        seen.add(path)
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Require lowercase SHA256 source image hashes")
        if source_split not in ("train", "val", "test"):
            raise ValueError("Unknown dataset split")
        if digest in hashes and hashes[digest] != source_split:
            raise ValueError("Dataset declares byte-identical images across splits")
        hashes[digest] = source_split
        if source_split == split:
            selected.append(record)
    if not selected:
        raise ValueError("Selected split is empty")
    if count is not None:
        if not 0 < count <= len(selected):
            raise ValueError("count must fit the selected split")
        selected = random.Random(seed).sample(selected, count)
    return selected


def repository_preprocess(metadata, model_args=None):
    """Resolve preprocessing through existing loaders without constructing the trained head."""
    from mini_trainer.modeling.architectures.load import get_dynamic_model, get_model, resolve_backbone_getter
    from mini_trainer.utils import string_to_dtype

    name = metadata["backbone_class"]
    getter, _ = resolve_backbone_getter(name)
    args = {} if getter is get_dynamic_model else {"pretrained": False, "local_files_only": True}
    args.update(model_args or {})
    args["resize_size"] = metadata["resize_size"]
    dtype = string_to_dtype(metadata.get("preprocess_dtype") or metadata.get("_dtype", "float32"))
    _, _, preprocess, _, _ = get_model(name, model_args=args, preprocess_dtype=dtype)
    return preprocess


def prepare(
    dataset_manifest,
    data_root,
    export_manifest,
    output,
    split,
    count=None,
    seed=42,
    batch_size=8,
    source_levels=None,
    level_names=None,
    classifier_module=None,
    preprocess_factory="dev.benchmarks.inference.prepare_inputs:repository_preprocess",
    factory_args=None,
    threads=1,
    score_semantics=None,
):
    import torch

    from mini_trainer.data import get_inference_dataloader

    if batch_size < 1 or threads < 1:
        raise ValueError("Require positive batch size and thread count")
    dataset_bytes, export_bytes = Path(dataset_manifest).read_bytes(), Path(export_manifest).read_bytes()
    dataset, exported = json.loads(dataset_bytes), json.loads(export_bytes)
    records = select_records(dataset, split, count, seed)
    heads = exported["classifiers"]
    if classifier_module is not None:
        heads = [head for head in heads if head["module"] == classifier_module]
    if len(heads) != 1:
        raise ValueError("Select one unambiguous classifier metadata entry with classifier_module")
    metadata = heads[0]["metadata"]
    classes = ordered_classes(metadata["cls2idx"])
    if score_semantics is None:
        raise ValueError("Declare logits/probabilities score semantics explicitly")
    semantics = [score_semantics] if isinstance(score_semantics, str) else list(score_semantics)
    if len(semantics) == 1:
        semantics *= len(classes)
    if len(semantics) != len(classes) or any(s not in ("logits", "probabilities") for s in semantics):
        raise ValueError("Supply score semantics for every exported level")
    dataset_classes = ordered_classes(dataset["class_spec"]["cls2idx"])
    source_levels = list(range(len(classes))) if source_levels is None else source_levels
    if (
        len(source_levels) != len(classes)
        or len(set(source_levels)) != len(source_levels)
        or any(type(i) is not int or not 0 <= i < len(dataset_classes) for i in source_levels)
    ):
        raise ValueError("Supply one distinct valid source level per exported classifier level")
    if [dataset_classes[i] for i in source_levels] != classes:
        raise ValueError("Export class ordering differs from the selected dataset levels")
    names = [f"level_{i}" for i in range(len(classes))] if level_names is None else level_names
    if len(names) != len(classes) or len(set(names)) != len(names) or any(not isinstance(n, str) or not n for n in names):
        raise ValueError("Supply unique names for every exported level")
    outputs = exported["outputs"]
    if len(outputs) != len(classes):
        raise ValueError("Export outputs do not match the selected classifier's levels")
    if exported["preprocessing"]["in_graph"]:
        raise ValueError("This preparation command expects preprocessing outside the graph")
    input_spec = exported["input"]
    if input_spec["dtype"] not in ("float16", "float32", "float64"):
        raise ValueError("This image preparation path requires a NumPy-compatible floating ONNX input")
    if "resize_size" not in metadata:
        raise ValueError("Export metadata must declare the repository image-reader resize_size")
    root = Path(data_root).resolve()
    paths = [(root / record["path"]).resolve() for record in records]
    if any(not path.is_relative_to(root) for path in paths):
        raise ValueError("Source images must remain under data_root")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": 1,
        "status": "running",
        "runner_sha256": file_hash(__file__),
        "dataset_manifest_sha256": hashlib.sha256(dataset_bytes).hexdigest(),
        "export_manifest_sha256": hashlib.sha256(export_bytes).hexdigest(),
        "export_source": exported.get("source"),
        "classifier_metadata": metadata,
        "selection": {
            "split": split,
            "count": count,
            "seed": seed,
            "method": "manifest order" if count is None else "random.Random(seed).sample",
        },
        "settings": {"batch_size": batch_size, "threads": threads, "workers": 0, "source_levels": source_levels},
        "source_records": records,
        "batches": [],
        "scope": "Explicit image preprocessing and input artifacts; not calibration, model inference or target-hardware acceptance.",
    }
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(threads)
        for path, record in zip(paths, records, strict=True):
            if file_hash(path) != record["sha256"]:
                raise ValueError(f"Source image hash mismatch: {path}")
        factory = preprocess_factory
        if isinstance(factory, str):
            module, name = factory.split(":", 1)
            factory = getattr(importlib.import_module(module), name)
        factory_source = inspect.getsourcefile(factory)
        report["preprocessing"] = {
            "factory": f"{factory.__module__}:{factory.__qualname__}",
            "factory_args": factory_args or {},
            "factory_source_sha256": file_hash(factory_source) if factory_source else None,
            "torch": torch.__version__,
            "numpy": np.__version__,
        }
        with seeded(seed):
            preprocess = factory(metadata, **(factory_args or {}))
        report["preprocessing"]["description"] = repr(preprocess)
        _, loader = get_inference_dataloader(
            images=[str(path) for path in paths],
            resize_size=metadata["resize_size"],
            batch_size=batch_size,
            num_workers=0,
            device="cpu",
            dtype=torch.float32,
        )
        samples = []
        for index, record in enumerate(records):
            targets = record["targets"]
            labels = []
            for classes_at_level, source_level in zip(classes, source_levels, strict=True):
                target = targets[source_level]
                if type(target) is not int or not 0 <= target < len(classes_at_level):
                    raise ValueError("Dataset target index is outside the declared class mapping")
                labels.append(classes_at_level[target])
            samples.append({"instance_id": index, "filename": record["path"], "labels": labels})
        offset = 0
        with torch.inference_mode(), seeded(seed):
            for index, images in enumerate(loader):
                transformed = preprocess(images)
                if not isinstance(transformed, torch.Tensor):
                    raise ValueError("Image preprocessing must return one tensor for the exported image input")
                array = transformed.detach().cpu().to(getattr(torch, input_spec["dtype"])).numpy()
                shape = input_spec["shape"]
                if array.ndim != len(shape) or any(type(n) is int and n != actual for n, actual in zip(shape, array.shape, strict=True)):
                    raise ValueError("Prepared tensor shape does not match the exported input")
                if len(array) != len(images) or not np.isfinite(array).all():
                    raise ValueError("Preprocessing must preserve batch identity and produce finite values")
                path = output / f"batch-{index:05d}.npz"
                np.savez(path, **{input_spec["name"]: array})
                ids = [str(i) for i in range(offset, offset + len(array))]
                report["batches"].append({"path": path.name, "sha256": file_hash(path), "sample_ids": ids})
                offset += len(array)
        if offset != len(records):
            raise ValueError("Preparation did not cover all selected records")
        for path, record in zip(paths, records, strict=True):
            if file_hash(path) != record["sha256"]:
                raise ValueError(f"Source image changed during preparation: {path}")
        manifest = {
            "schema_version": 1,
            "split": split,
            "provenance": {
                key: report[key]
                for key in (
                    "dataset_manifest_sha256",
                    "export_manifest_sha256",
                    "export_source",
                    "selection",
                    "preprocessing",
                    "runner_sha256",
                )
            },
            "batch_input": input_spec["name"],
            "batches": report["batches"],
            "levels": [
                {"name": name, "classes": cls, "output": binding["name"], "score_semantics": semantics[i]}
                for i, (name, cls, binding) in enumerate(zip(names, classes, outputs, strict=True))
            ],
            "samples": samples,
        }
        manifest_path = output / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        (calibration_manifest if split == "train" else inference_manifest)(manifest_path)
        report.update(status="prepared", manifest_sha256=file_hash(manifest_path))
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        torch.set_num_threads(previous_threads)
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--export-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], required=True)
    parser.add_argument("--count", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--source-levels", type=int, nargs="+")
    parser.add_argument("--level-names", nargs="+")
    parser.add_argument("--classifier-module")
    parser.add_argument("--preprocess-factory", default="dev.benchmarks.inference.prepare_inputs:repository_preprocess")
    parser.add_argument("--factory-args", type=json.loads)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--score-semantics", choices=["logits", "probabilities"], nargs="+", required=True)
    prepare(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
