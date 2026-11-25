"""Build a release folder from a checkpoint: the trunk and each head as safetensors, plus ONNX graphs.

The cut is the classifier's input. Everything before it is the **trunk**; the classifier is the **head**
(its hidden layer included), because a head trained on a frozen trunk (``mt_train --freeze-backbone``)
trains its own hidden layer. The trunk graph also returns the default head's normalised embedding, so
it doubles as the embedding model. Layout of a release:

    config.json                 preprocessing, trunk, heads, files and their hashes
    trunk.safetensors           trunk weights (timm names for timm backbones)
    trunk.onnx, trunk_fp16.onnx uint8 square image -> features, embedding (fp16: half-precision trunk)
    heads/<name>.safetensors    hidden layer, prototypes, magnitudes, biases, the tree
    heads/<name>.json           ranks and labels
    model.safetensors           trunk and default head in one file (``trunk.*`` and ``head.*`` tensors)
    model.onnx (+ .onnx.data)   trunk and default head in one graph -> embedding, logits per rank

safetensors is the release format (PyTorch and Hugging Face); ONNX is the deployment format.
"""

import copy
import gc
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from mini_trainer.hierarchical.model import HierarchicalClassifier
from mini_trainer.modeling import Classifier, EmbeddingContext, classification_module
from mini_trainer.modeling.architectures import ResampleNormalize
from mini_trainer.modeling.architectures.load import get_dynamic_model, resolve_backbone_getter
from mini_trainer.modeling.quantized_training import load_training_weights

FORMAT = "mini_trainer-release-v1"
SINGLE_FILE_LIMIT = 1_800_000_000  # bytes; protobuf caps an ONNX file at 2 GiB, larger graphs keep weights in .onnx.data


def load_checkpoint(weights: str | Path):
    """The model and its evaluation preprocessing, through mini_trainer's own loader."""
    state = load_training_weights(str(weights), map_location="cpu")
    state = state.get("model", state)
    getter, _ = resolve_backbone_getter(Classifier.extract_metadata(state)["backbone_class"])
    model_args = {} if getter is get_dynamic_model else {"pretrained": False}
    model, preprocess = Classifier.build(weights=state, device="cpu", dtype=torch.float32, model_args=model_args)
    del state  # the model holds its own copy
    return model.eval(), preprocess


def split(model: nn.Module) -> tuple[nn.Module, Classifier]:
    """Detach the classifier: the model then computes the classifier's input (the trunk features)."""
    classifier = classification_module(model)
    if type(classifier) not in (Classifier, HierarchicalClassifier):
        raise NotImplementedError(f"{type(classifier).__name__}: releases support flat and leaf-aggregated hierarchical heads")
    if not classifier.normalized or classifier.active_indices is not None:
        raise NotImplementedError("Releases need a normalised (cosine) head without an active class mask")
    model.set_submodule(model._backbone_output_name, nn.Identity())
    return model, classifier


def preprocessing(preprocess, classifier: Classifier) -> dict:
    """The evaluation preprocessing as data: a square read outside the graph, the rest inside it."""
    meta = classifier.metadata
    if meta.get("resize_method") != "crop" or not isinstance(preprocess.transform, ResampleNormalize):
        raise NotImplementedError("Releases need square-crop preprocessing (resize_method='crop')")
    transform = preprocess.transform
    return {
        "reader": "RGB; largest centred square (fastai's rounding); PIL bilinear resize to read_size; uint8 CHW; EXIF ignored",
        "read_size": transform.read_size,
        "model_size": transform.size,
        "in_graph": "resample read_size -> model_size (identity grid, bilinear, reflection, align_corners=True); /255; mean/std",
        "mean": list(transform.mean),
        "std": list(transform.std),
    }


def trunk_state(trunk: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.contiguous() for key, value in trunk.state_dict().items() if isinstance(value, torch.Tensor)}


def tensor_digest(tensors: dict[str, torch.Tensor]) -> str:
    """sha256 over names, dtypes, shapes and bytes: equal digests mean identical weights."""
    digest = hashlib.sha256()
    for key in sorted(tensors):
        value = tensors[key].detach().cpu().contiguous()
        digest.update(f"{key}|{value.dtype}|{tuple(value.shape)}|".encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())  # reshape: 0-d tensors (BatchNorm counters)
    return digest.hexdigest()


def head_tensors(classifier: Classifier) -> dict[str, torch.Tensor]:
    """What :func:`standalone.score` needs: unit prototypes, magnitudes, biases, hidden layer, the tree.

    A shared ``global_scale`` is written as every class's magnitude (scored magnitude-after-angle).
    """
    weight = classifier.linear.parametrizations.weight
    scale = getattr(classifier, "global_scale", None)
    tensors = {
        "direction": F.normalize(weight.original1, 2, 1),
        "magnitude": weight.original0.squeeze(1) if scale is None else scale.expand(weight.original1.shape[0]),
        "bias": classifier.linear.bias,
    }
    if classifier.hidden:
        tensors["hidden.weight"], tensors["hidden.bias"] = classifier.hidden.weight, classifier.hidden.bias
    for k in range(getattr(classifier, "num_masks", 0)):
        tensors[f"parent.{k}"] = getattr(classifier, f"mask_{k}")
    return {key: value.detach().contiguous() for key, value in tensors.items()}


def head_labels(classifier: Classifier, ranks: list[str]) -> list[list[str]]:
    """Labels in output order, per rank (finest first), from the checkpoint's class mappings."""
    mapping = classifier.metadata["cls2idx"]
    mappings = [mapping[str(k)] for k in range(len(ranks))] if isinstance(classifier, HierarchicalClassifier) else [mapping]
    labels = [sorted(m, key=m.get) for m in mappings]
    if any([m[name] for name in names] != list(range(len(names))) for m, names in zip(mappings, labels)):
        raise ValueError("Class indices are not contiguous from 0")
    return labels


def write_head(folder: Path, name: str, classifier: Classifier, ranks: list[str] | None, source: str) -> dict:
    from safetensors.torch import save_file

    n_ranks = getattr(classifier, "num_masks", 0) + 1
    ranks = ranks or [f"rank{k}" for k in range(n_ranks)]
    if len(ranks) != n_ranks:
        raise ValueError(f"The head has {n_ranks} ranks; got names {ranks}")
    tensors = head_tensors(classifier)
    labels = head_labels(classifier, ranks)
    sizes = [len(names) for names in labels]
    meta = {
        "format": FORMAT,
        "ranks": ranks,
        "sizes": sizes,
        "features": classifier.in_features,
        "embedding": classifier.preclassification_size,
        "magnitude_after_angle": bool(
            getattr(classifier, "magnitude_after_angle", False) or getattr(classifier, "global_scale", None) is not None
        ),
        "formula": "logit = sqrt(D-2)*asin(clamp(g*cos, -1+1e-7, 1-1e-7)) + b; coarser rank = logsumexp of children",
        "source": source,
    }
    (folder / "heads").mkdir(exist_ok=True)
    save_file(tensors, str(folder / f"heads/{name}.safetensors"), metadata={"format": "pt", "mini_trainer_head": json.dumps(meta)})
    (folder / f"heads/{name}.json").write_text(json.dumps({"ranks": ranks, "labels": labels}) + "\n")
    return {"file": f"heads/{name}.safetensors", "labels": f"heads/{name}.json", **{k: meta[k] for k in ("ranks", "sizes", "embedding")}}


class ReleaseGraph(nn.Module):
    """uint8 square images -> (features, embedding), or with ``logits`` -> (embedding, logits per rank)."""

    def __init__(self, trunk: nn.Module, classifier: Classifier, preprocess, logits: bool = False, half: bool = False):
        super().__init__()
        self.trunk, self.classifier, self.logits, self.half = trunk, classifier, logits, half
        self.preprocess = preprocess  # a plain callable: the checkpoint's own ResampleNormalize (and dtype step)

    def forward(self, images):
        x = self.preprocess(images)
        features = self.trunk(x.half() if self.half else x).float()
        if not self.logits:
            return features, self.classifier.preclassification(features)
        EmbeddingContext.activate()  # the embedding of this very forward, as mt_predict collects it
        try:
            outputs = self.classifier(features)
            embedding = EmbeddingContext.get()
        finally:
            EmbeddingContext.clear()
        return (embedding, *(outputs if isinstance(outputs, (list, tuple)) else [outputs]))


def export_graph(graph: ReleaseGraph, read_size: int, path: Path, names: list[str]) -> list[str]:
    """Dynamo ONNX export with a dynamic batch; returns the written file names.

    Weights go to ``<name>.onnx.data`` only when the graph is too large for one protobuf file.
    """
    import onnx

    example = torch.zeros(2, 3, read_size, read_size, dtype=torch.uint8)  # batch one would be specialised
    graph.eval()
    with torch.no_grad():
        graph(example)  # warm the head's lazy caches eagerly, so tracing reads plain buffers
        torch.onnx.export(
            graph,
            (example,),
            str(path),
            input_names=["images"],
            output_names=names,
            opset_version=18,
            dynamo=True,
            dynamic_shapes=({0: torch.export.Dim("batch", min=1)},),
            external_data=True,
        )
    data = path.with_name(path.name + ".data")
    if data.exists() and data.stat().st_size < SINGLE_FILE_LIMIT:
        proto = onnx.load(str(path))  # reads the external weights
        data.unlink()
        onnx.save(proto, str(path))
    onnx.checker.check_model(str(path))
    return [p.name for p in (path, data) if p.exists()]


def write_combined(folder: Path, head: dict) -> None:
    """``model.safetensors``: the trunk and the default head in one file, tensors copied from the separate files."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    tensors, metadata = {}, {"format": "pt"}
    for prefix, file in (("trunk.", "trunk.safetensors"), ("head.", head["file"])):
        with safe_open(str(folder / file), framework="pt") as handle:
            metadata.update({k: v for k, v in handle.metadata().items() if k != "format"})
            tensors.update({prefix + key: handle.get_tensor(key) for key in handle.keys()})
    save_file(tensors, str(folder / "model.safetensors"), metadata=metadata)


def file_table(folder: Path) -> dict:
    files = {}
    for path in sorted(p for p in folder.rglob("*") if p.is_file() and p.name != "config.json"):
        with path.open("rb") as handle:
            files[path.relative_to(folder).as_posix()] = {
                "size": path.stat().st_size,
                "sha256": hashlib.file_digest(handle, "sha256").hexdigest(),
            }
    return files


def write_config(folder: Path, config: dict) -> None:
    config["files"] = file_table(folder)
    (folder / "config.json").write_text(json.dumps(config, indent=2) + "\n")


def export_release(
    weights: str | Path,
    output: str | Path,
    ranks: list[str] | None = None,
    head_name: str = "main",
    fp16: bool = True,
    combined: bool = True,
) -> Path:
    """Checkpoint -> release folder (see the module docstring). Verify it with :func:`verify_release`."""
    from safetensors.torch import save_file

    folder = Path(output)
    if folder.exists():
        raise FileExistsError(f"Release folder already exists: {folder}")
    folder.mkdir(parents=True)
    with Path(weights).open("rb") as handle:
        checkpoint_sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
    model, preprocess = load_checkpoint(weights)
    trunk, classifier = split(model)
    prep = preprocessing(preprocess, classifier)
    state = trunk_state(trunk)
    digest = tensor_digest(state)
    save_file(state, str(folder / "trunk.safetensors"), metadata={"format": "pt", "trunk_sha256": digest})
    del state
    head = write_head(folder, head_name, classifier, ranks, Path(weights).name)

    print("exporting trunk.onnx", flush=True)
    export_graph(ReleaseGraph(trunk, classifier, preprocess), prep["read_size"], folder / "trunk.onnx", ["features", "embedding"])
    if fp16:
        print("exporting trunk_fp16.onnx", flush=True)
        half = ReleaseGraph(copy.deepcopy(trunk).half(), classifier, preprocess, half=True)
        export_graph(half, prep["read_size"], folder / "trunk_fp16.onnx", ["features", "embedding"])
        del half
    if combined:
        write_combined(folder, head)
        print("exporting model.onnx (trunk and head in one graph)", flush=True)
        names = ["embedding"] + [f"logits_{rank}" for rank in head["ranks"]]
        model_files = export_graph(
            ReleaseGraph(trunk, classifier, preprocess, logits=True), prep["read_size"], folder / "model.onnx", names
        )

    config = {
        "format": FORMAT,
        "source": {"checkpoint": Path(weights).name, "checkpoint_sha256": checkpoint_sha256},
        "preprocessing": prep,
        "trunk": {
            "backbone": classifier.metadata["backbone_class"],
            "features": classifier.in_features,
            "sha256": digest,
            "safetensors": "trunk.safetensors",
            "onnx": "trunk.onnx",
            **({"onnx_fp16": "trunk_fp16.onnx"} if fp16 else {}),
            "inputs": {"images": ["batch", 3, prep["read_size"], prep["read_size"], "uint8"]},
            "outputs": {"features": "classifier input", "embedding": f"the {head_name} head's L2-normalised embedding"},
        },
        "heads": {head_name: head},
        "default_head": head_name,
        **({"combined": {"safetensors": "model.safetensors", "onnx": model_files, "outputs": names}} if combined else {}),
    }
    write_config(folder, config)
    del model, trunk, classifier
    gc.collect()  # the traced graphs hold reference cycles; free the weights before verification loads them again
    return folder


def add_head(weights: str | Path, release: str | Path, name: str, ranks: list[str] | None = None) -> Path:
    """Add a head trained on this release's frozen trunk; refuses a checkpoint whose trunk differs."""
    folder = Path(release)
    config = json.loads((folder / "config.json").read_text())
    if name in config["heads"]:
        raise FileExistsError(f"Head {name!r} already exists in {folder}")
    model, preprocess = load_checkpoint(weights)
    trunk, classifier = split(model)
    digest = tensor_digest(trunk_state(trunk))
    if digest != config["trunk"]["sha256"]:
        raise ValueError(f"{Path(weights).name} has a different trunk (fine-tuned or another model): export it as its own release")
    if preprocessing(preprocess, classifier) != config["preprocessing"]:
        raise ValueError("The head was trained with different preprocessing than this release's trunk")
    config["heads"][name] = write_head(folder, name, classifier, ranks, Path(weights).name)
    write_config(folder, config)
    return folder
