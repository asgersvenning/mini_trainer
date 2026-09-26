import hashlib
import json
import os
from argparse import ArgumentParser
from typing import Any

import torch

from mini_trainer import get_logger
from mini_trainer.builders import BaseBuilder
from mini_trainer.config import (
    Formatter,
    defaults_from_function,
    dump_resolved_config,
    load_yaml_config,
    merge_dicts,
    restructure_cli_args,
)
from mini_trainer.data import auto_find_images, get_metadata
from mini_trainer.logging import BaseResultCollector, RawResultCollector
from mini_trainer.modeling import EmbeddingContext, classification_module, predict
from mini_trainer.modeling.mask import restrict_class_labels
from mini_trainer.utils import TQDM, increment_name_dir, setup_device


def main(  # noqa: D417
    input: str,
    weights: str,
    output: str = ".",
    name: str | None = None,
    threshold: float = 0,
    data_index: str | None = None,
    class_spec: str | dict | None = None,
    subsample: int | None = None,
    device: str | torch.device = "cuda",
    dtype: str | torch.dtype = "float16",
    builder: type[BaseBuilder] = BaseBuilder,
    collector_cls: type[BaseResultCollector] = BaseResultCollector,
    spec_model_dataloader_kwargs: dict[str, Any] = {},
    model_builder_kwargs: dict[str, Any] = {},
    dataloader_builder_kwargs: dict[str, Any] = {"batch_size": 64},
    augmentation_builder_kwargs: dict[str, Any] = {},
    criterion_builder_kwargs: dict[str, Any] = {},
    collector_cls_kwargs: dict[str, Any] = {},
    class_list: str | None = None,
):
    """Run inference and save collector output under ``output/name``.

    The default collector writes ``mini_metric.csv``; ``RawResultCollector``
    writes logits, labels and paths to ``predictions.pt``. Configuration is saved
    alongside results. Only model, dataloader and collector keyword groups are
    consumed; the other builder keyword groups remain for API compatibility.

    Args:
        input: Image, image directory or gbifxdl Parquet. Labelled directories and
            Parquet retain supplied test splits; mixed/root-level images are unlabelled.
        weights: Trained model weights.
        output: Parent directory for results.
        name: Subdirectory name, sanitized and incremented to avoid collisions.
            ``None`` uses the legacy API name ``train``; the CLI defaults to ``predict``.
        threshold: Acceptance threshold recorded by the default CSV collector.
        data_index: Metadata with path, label and split arrays. When supplied,
            its test rows replace discovery from ``input``.
        class_spec: Existing JSON file or dictionary of model-construction metadata.
        subsample: Keep every Nth selected observation (2 keeps half).
        device: Inference device.
        dtype: CUDA autocast dtype and raw-logit output dtype. The default builder
            loads float32 model parameters; the default loader supplies uint8 images.
        builder: Supplies model/preprocessing and inference-loader hooks.
        collector_cls: Collects batches and saves predictions.
        class_list: UTF-8 file with one candidate label per line (leaf labels for
            hierarchical models). Blank/duplicate lines are ignored; input images
            and ground truth are retained independently of candidate filtering.
    """
    orig_args = locals()
    if name is None:
        name = "train"
    name = increment_name_dir(name, output)

    input = os.path.abspath(input)
    output_dir = None if output is None else os.path.abspath(os.path.join(output, name))
    if output_dir is not None:
        try:
            os.makedirs(output_dir, exist_ok=False)
        except OSError as e:
            e.add_note(f"Training output directory already exists: {output_dir}. Perhaps a run of with the `{{name=}}` already exists?")

    device = setup_device(device)
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype.removeprefix("torch.").strip().lower())
        assert isinstance(dtype, torch.dtype)

    dump_resolved_config(
        output_dir=output_dir,
        fn=main,
        local_vars=orig_args,
        overrides={
            "input": input,
            "output": output_dir,
            "device": device,
            "dtype": dtype,
            "name": name,
        },
        verbose=collector_cls_kwargs.get("verbose", False),
    )

    if class_spec is None:
        class_spec = {}
    if isinstance(class_spec, str):
        with open(class_spec) as f:
            class_spec = json.load(f)
        assert isinstance(class_spec, dict)

    model_dtype = torch.float32
    nn_model, model_preprocess = builder.build_model(
        weights=weights, device=device, dtype=model_dtype, strict=False, **{**class_spec, **model_builder_kwargs}
    )
    nn_model.eval()
    metadata = classification_module(nn_model).metadata.copy()
    if class_list is not None:
        with open(class_list, "rb") as handle:
            contents = handle.read()
        labels_to_keep = [line.strip() for line in contents.decode("utf-8-sig").splitlines() if line.strip()]
        report = restrict_class_labels(nn_model, labels_to_keep)
        report.update(source=os.path.abspath(class_list), sha256=hashlib.sha256(contents).hexdigest())
        with open(os.path.join(output_dir, "class_filter.json"), "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        get_logger().info(
            f"Class list: retained {report['retained_count']}/{report['original_candidate_count']} candidates; "
            f"{len(report['missing_labels'])} requested labels absent from active vocabulary",
        )
        # Keep original metadata for resolving ground truth, including excluded labels.

    labels: list[int] | list[list[int]] | None = None
    if data_index is not None:
        _data_metadata = get_metadata(data_index, **metadata)
        images: list[str] = [p for p, s in zip(_data_metadata["path"], _data_metadata["split"]) if s == "test"]
        labels = [p for p, s in zip(_data_metadata["label"], _data_metadata["split"]) if s == "test"]
    else:
        labels, images = auto_find_images(input, **metadata)
    if subsample is not None and subsample > 1:
        labels, images = labels[::subsample], images[::subsample]

    loader = builder.build_inference_dataloader(images=images, device=device, dtype=dtype, **{**metadata, **dataloader_builder_kwargs})
    if not isinstance(loader, torch.utils.data.DataLoader):
        raise TypeError(
            f"Expected `dataloader_builder` to return an objectsinheriting from `torch.utils.data.DataLoader`, but got `{type(loader)}."
        )

    collector = collector_cls(model=nn_model, **collector_cls_kwargs)

    idx = 0
    for batch in TQDM(loader, desc="Running inference", unit="batch"):
        with torch.inference_mode():
            batch = model_preprocess(batch)
            batch = batch.to(device, non_blocking=True)
            with (
                torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype != torch.float32 and device.type == "cuda"),
                EmbeddingContext(),
            ):
                if not isinstance(collector, RawResultCollector):
                    predictions = predict(nn_model, batch)
                else:
                    predictions = nn_model(batch)
                    if isinstance(predictions, torch.Tensor):
                        predictions = predictions.to(dtype=dtype)
                    else:
                        predictions = [p.to(dtype=dtype) for p in predictions]
                idxs = slice(idx, idx + len(batch))
                idx += len(batch)
                collector.collect(
                    paths=images[idxs],
                    predictions=predictions,  # type: ignore
                    labels=None if labels is None else labels[idxs],  # type: ignore
                )
    del loader, nn_model

    collector.save(os.path.join(output, name), threshold=threshold)


def cli(description="Classify images with a trained model", **extra_kwargs):  # noqa: D103
    parser = ArgumentParser(prog="predict", description=description, formatter_class=Formatter)

    cfg_args = parser.add_argument_group("Config [optional]")
    cfg_args.add_argument(
        "--config",
        type=str,
        default=None,
        required=False,
        help="Path to a YAML config file; values act as defaults and are overridden by explicit CLI flags.",
    )

    if extra_kwargs:
        for argname, kwargs in extra_kwargs.items():
            args = []
            if None in kwargs:
                args = kwargs.pop(None)
                if not hasattr(args, "__iter__") or isinstance(args, str):
                    args = [args]
                else:
                    args = list(args)
            args.insert(0, f"--{argname}")
            parser.add_argument(*args, **kwargs)

    input_args = parser.add_argument_group("Input [mandatory]")
    input_args.add_argument(
        "-i",
        "--input",
        type=str,
        default=None,
        required=False,
        help="Image file, image directory or gbifxdl Parquet; supplied test splits are retained.",
    )
    mod_args = parser.add_argument_group("Model [optional]")
    mod_args.add_argument(
        "--class-list",
        "--class_list",
        dest="class_list",
        default=None,
        help="File with one candidate label per line (species IDs for hierarchical models). "
        "Restricts predictions only; all input images and ground-truth labels are retained.",
    )
    mod_args.add_argument(
        "-w",
        "--weights",
        type=str,
        dest="weights",
        default=None,
        required=False,
        help="Trained model weights.",
    )
    out_args = parser.add_argument_group("Output [optional]")
    out_args.add_argument(
        "-o",
        "--output",
        type=str,
        default=".",
        required=False,
        help='Parent directory for prediction results (default=".").',
    )
    out_args.add_argument(
        "-n", "--name", type=str, default="predict", required=False, help='Name of the output predictions.\nDefault is "predict".'
    )
    inf_args = parser.add_argument_group("Inference [optional]")
    inf_args.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=None,
        required=False,
        help="Acceptance threshold for mini_metrics CSV output (default=0, always predict).",
    )
    inf_args.add_argument(
        "-r",
        "--raw",
        action="store_true",
        default=False,
        required=False,
        help="Save raw logits, labels and paths to predictions.pt instead of prediction CSV.",
    )
    inf_args.add_argument(
        "-D",
        "--data_index",
        type=str,
        default=None,
        required=False,
        help='Metadata with "path", "label" and "split" arrays; only test rows are used. Overrides input discovery.',
    )
    mod_args.add_argument(
        "-C",
        "--class_spec",
        type=str,
        default=None,
        required=False,
        help="Existing JSON file with class mappings and other model-construction metadata.",
    )
    inf_args.add_argument(
        "--batch_size",
        type=int,
        dest="dataloader_builder_kwargs.batch_size",
        default=None,
        required=False,
        help="Images per inference batch (default=64).",
    )
    cfg_args = parser.add_argument_group("Runtime [optional]")
    cfg_args.add_argument(
        "--cuda-prefetch",
        action="store_true",
        dest="dataloader_builder_kwargs.cuda_prefetch",
        help="Stage one CPU batch ahead on a CUDA transfer stream (opt-in; uses extra device memory).",
    )
    cfg_args.add_argument(
        "--subsample",
        type=int,
        default=None,
        required=False,
        help="Keep every Nth selected image (default: all images).",
    )
    cfg_args.add_argument("--device", type=str, default=None, required=False, help='Inference device (default="cuda").')
    cfg_args.add_argument(
        "--dtype",
        type=str,
        default=None,
        required=False,
        help="CUDA autocast and raw-logit output dtype (default=float16); default model parameters remain float32.",
    )
    cfg_args.add_argument(
        "--num_workers",
        type=int,
        dest="dataloader_builder_kwargs.num_workers",
        default=None,
        required=False,
        help="DataLoader workers. Default: 0-32 based on CPUs available to this process; use 0 to load in the main process.",
    )
    cfg_args.add_argument(
        "--seed",
        type=int,
        default=None,
        required=False,
        help="Unsupported: the prediction API does not accept a seed.",
    )
    cfg_args.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        dest="collector_cls_kwargs.verbose",
        default=None,
        required=False,
        help="Print the resolved configuration.",
    )
    cli_args = vars(parser.parse_args())

    if cli_args.pop("raw", False):
        cli_args["collector_cls"] = RawResultCollector

    defaults_full = defaults_from_function(main)
    config_full = {k: v for k, v in load_yaml_config(cli_args.pop("config")).items() if k in defaults_full}
    cli_full = restructure_cli_args(cli_args)

    args = merge_dicts(defaults_full, config_full, cli_full)

    if args.get("input") is None:
        raise SystemExit("error: the following arguments are required: --input (via CLI or config)")

    return args


def run():  # noqa: D103
    main(**cli())


if __name__ == "__main__":
    run()
