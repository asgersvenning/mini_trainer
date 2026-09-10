"""Replay one saved checkpoint's validation batches; diagnostic, not a benchmark."""

import argparse
import hashlib
import json
from pathlib import Path

from compare import digest, write_json


def replay(config, data, spec, checkpoint, *, precision, prefetch, output, device="cuda"):
    import torch

    from mini_trainer.data.loader import get_dataset_dataloader
    from mini_trainer.hierarchical.integration import HierarchicalBuilder
    from mini_trainer.modeling import Classifier

    device = torch.device(device)
    dtype = torch.float16 if precision == "fp16" else torch.float32
    model, preprocess = Classifier.build(weights=checkpoint, device=device, dtype=torch.float32, model_args={"pretrained": False})
    model.eval()
    parameters_finite = all(bool(torch.isfinite(p).all()) for p in model.parameters())
    nonfinite_buffers = [name for name, value in model.named_buffers() if not bool(torch.isfinite(value).all())]
    labels = [label for label, split in zip(data["class"], data["split"], strict=True) if split == "train"]
    criterion = HierarchicalBuilder.build_criterion(
        num_classes=spec["num_classes"],
        labels=labels,
        device=device,
        dtype=dtype,
        weighted=True,
        label_smoothing=None,
        weights=[1, 1, 1],
    )
    selected = [i for i, split in enumerate(data["split"]) if split == "validation"]
    metadata = {key: [data[key][i] for i in selected] for key in ("path", "class")}
    _, loaders = get_dataset_dataloader(
        metadata,
        resize_size=(config["size"], config["size"]),
        modes=("val",),
        batch_size=config["global_batch_size"],
        num_workers=config["num_workers_per_rank"],
        device=device,
        dtype=dtype,
        cache=None,
        multilabel=True,
        cuda_prefetch=prefetch,
    )
    records = []
    offset = 0
    with (output / "batches.jsonl").open("w") as handle:
        for index, (batch, targets) in enumerate(loaders[0]):
            batch = batch.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with torch.inference_mode(), torch.autocast(device.type, dtype=dtype, enabled=precision == "fp16"):
                images = preprocess(batch)
                logits = model(images)
                losses = criterion(logits, targets)
            # Inspect only after inference has consumed the asynchronous batch.
            # These synchronizations can still hide timing-sensitive failures.
            finite = {
                "input": bool(torch.isfinite(images).all()),
                "outputs": [bool(torch.isfinite(value).all()) for value in logits],
                "losses": [bool(torch.isfinite(value).all()) for value in losses],
            }
            values = [float(value) for value in losses]
            record = {
                "batch": index,
                "paths": metadata["path"][offset : offset + len(batch)],
                "input_sha256": hashlib.sha256(batch.cpu().contiguous().numpy().tobytes()).hexdigest(),
                "targets_sha256": hashlib.sha256(targets.cpu().contiguous().numpy().tobytes()).hexdigest(),
                "finite": finite,
                "losses": [value if good else str(value) for value, good in zip(values, finite["losses"], strict=True)],
            }
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            records.append(record)
            offset += len(batch)
            print(f"batch={index} finite={finite}", flush=True)
    passed = (
        bool(records)
        and parameters_finite
        and not nonfinite_buffers
        and all(r["finite"]["input"] and all(r["finite"]["outputs"]) and all(r["finite"]["losses"]) for r in records)
    )
    result = {
        "status": "passed" if passed else "nonfinite",
        "precision": precision,
        "prefetch": prefetch,
        "parameters_finite": parameters_finite,
        "nonfinite_buffers": nonfinite_buffers,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": str(device),
        "preprocessing": repr(preprocess),
        "criterion": repr(criterion),
        "batches": len(records),
        "images": offset,
        "checkpoint_sha256": digest(checkpoint),
        "scope": "Reloaded checkpoint validation with per-batch synchronization; no optimizer, training or timing comparison",
    }
    write_json(output / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comparison", type=Path)
    parser.add_argument("run")
    parser.add_argument("--precision", choices=("fp16", "fp32"), required=True)
    parser.add_argument("--prefetch", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.comparison.resolve()
    config = json.loads((root / "comparison.json").read_text())
    if config["gpus"] != 1:
        raise ValueError("This diagnostic requires the single-GPU qualification")
    result = json.loads((root / "runs" / args.run / "result.json").read_text())
    checkpoint = result["checkpoint"]
    if digest(checkpoint) != result["checkpoint_sha256"]:
        raise ValueError("Checkpoint changed")
    prepared = json.loads((root / "prepared.json").read_text())
    for name in ("data_index.json", "class_spec.json"):
        if digest(root / name) != prepared[name]:
            raise ValueError(f"Frozen artifact changed: {name}")
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "source.json",
        {"comparison": str(root), "config_sha256": digest(root / "comparison.json"), "run": args.run, "checkpoint": checkpoint},
    )
    replay(
        config,
        json.loads((root / "data_index.json").read_text()),
        json.loads((root / "class_spec.json").read_text()),
        checkpoint,
        precision=args.precision,
        prefetch=args.prefetch,
        output=args.output,
    )


if __name__ == "__main__":
    main()
