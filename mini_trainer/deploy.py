"""Native PyTorch deployment. No dependency on the portable deployment package."""

import hashlib
import json
import os
import re
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from mini_trainer.builders import BaseBuilder
from mini_trainer.data.io import make_read_and_resize_fn
from mini_trainer.hierarchical.model import HierarchicalPrediction
from mini_trainer.modeling import Classifier, Prediction, classification_module, set_classification_mask
from mini_trainer.modeling.architectures.load import get_dynamic_model, resolve_backbone_getter
from mini_trainer.modeling.classifier import bypass_submodule
from mini_trainer.modeling.quantized_training import load_training_weights


@lru_cache(maxsize=1)
def _release():
    return json.loads(Path(__file__).with_name("nemo.json").read_text())


def _checkpoint_digest(state):
    """Identity of tensor contents and class order, independent of torch.save layout."""
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        if isinstance(value, torch.Tensor) and not key.endswith(".active_indices"):
            value = value.detach().cpu().contiguous()
            digest.update(json.dumps([key, str(value.dtype), list(value.shape)]).encode())
            digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    config = Classifier.extract_metadata(state)
    identity = {
        key: config.get(key)
        for key in ("backbone_class", "backbone_output_name", "classifier_class", "resize_size", "cls2idx", "normalized", "hidden")
    }
    digest.update(json.dumps(identity, sort_keys=True).encode())
    return digest.hexdigest()


def ensure_weights(model=None, weight_dir=None):
    """Resolve Nemo or a local checkpoint, preserving the legacy two-value return."""
    if model is not None and Path(str(model)).suffix in (".pt", ".pth"):
        path = Path(model).expanduser()
        if not path.is_file():
            raise FileNotFoundError(path)
        return str(path), str(path)
    release = _release()
    cache = Path(weight_dir or os.environ.get("MAMBO_CACHE", Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "mambo"))
    path = cache / "blobs" / release["sha256"]
    if not path.exists():
        if os.environ.get("MAMBO_OFFLINE") == "1":
            raise FileNotFoundError(f"Nemo checkpoint is not cached: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(release["url"], str(path), hash_prefix=release["sha256"])
    with path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != release["sha256"]:
            raise ValueError(f"Cached artifact integrity mismatch: {path}")
    return str(path), release["url"]


class TorchPreprocess:
    """Finish a batch of uint8 squares with native operations on its Torch device."""

    def __init__(self, torch, device):
        self.torch = torch
        recipe = _release()["preprocessing"]
        self.size, self.resized = recipe["square_size"], recipe["resize_size"]
        mean = torch.as_tensor(recipe["mean"], device=device)[:, None, None]
        std = torch.as_tensor(recipe["std"], device=device)[:, None, None]
        self.scale, self.bias = 1 / (255 * std), -mean / std

    def __call__(self, images):
        torch = self.torch
        images = torch.as_tensor(images, device=self.scale.device)
        single = images.ndim == 3
        if single:
            images = images.unsqueeze(0)
        if images.dtype != torch.uint8:
            if not images.is_floating_point() or not torch.isfinite(images).all() or images.min() < 0 or images.max() > 1:
                raise ValueError("Image arrays must be uint8 or finite floating point in [0,1]")
            images = (images * 255).round().to(torch.uint8)
        if images.shape[-2:] != (self.size, self.size):
            images = torch.nn.functional.interpolate(images, size=(self.size, self.size), mode="nearest")
        with torch.autocast(images.device.type, enabled=False):
            values = torch.nn.functional.interpolate(
                images.float(), size=(self.resized, self.resized), mode="bilinear", align_corners=False, antialias=False
            )
            start = (self.resized - self.size) // 2
            values = values[..., start : start + self.size, start : start + self.size]
            # One broadcast normalization kernel also produces compact NCHW storage.
            result = torch.addcmul(self.bias, values.round_(), self.scale)
            return result[0] if single else result


class _Prediction(HierarchicalPrediction):
    def _process(self, raw_prediction):
        if self.topk < 1 or self.topk > min(rank.shape[-1] for rank in raw_prediction):
            raise ValueError("topk must fit every retained rank")
        indices = [rank.argsort(dim=-1, descending=True, stable=True)[:, : self.topk] for rank in raw_prediction]
        return torch.stack([rank.gather(1, index) for rank, index in zip(raw_prediction, indices)], dim=-1), torch.stack(indices, dim=-1)

    def _extract_confidence(self, raw_prediction):
        # Release outputs are logits even when their values happen to sum to one.
        return torch.stack([rank.softmax(-1).gather(1, self.indices[..., i]) for i, rank in enumerate(raw_prediction)], dim=-1)


class Predictor:
    """Eager native predictor; CUDA and European scope retain legacy defaults."""

    def __init__(self, device="cuda", model=None, weights=None, class_mask=None, *, weight_dir=None, precision="auto"):
        if model is not None and weights is not None:
            raise ValueError("model and weights are mutually exclusive")
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("device must be cpu or cuda")
        if precision not in ("auto", "fp32", "fp16", "bf16"):
            raise ValueError("precision must be auto, fp32, fp16 or bf16")
        self.precision = ("fp16" if self.device.type == "cuda" else "fp32") if precision == "auto" else precision
        if self.device.type == "cpu" and self.precision != "fp32":
            raise ValueError("CPU inference requires auto or fp32 precision")
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("PyTorch CUDA is unavailable; choose device='cpu' or configure CUDA")
            if self.precision == "bf16":
                with torch.cuda.device(self.device):
                    if not torch.cuda.is_bf16_supported(including_emulation=False):
                        raise RuntimeError("BF16 requires native device support; choose fp16 or fp32")
        local = weights is not None or (model is not None and Path(str(model)).suffix in (".pt", ".pth"))
        release = _release()
        self.preset = "custom" if local else self._preset_name(model or "europe", release["presets"])
        self.source = None
        if weights is None:
            weights, self.source = ensure_weights(model, weight_dir)
        self.weights = weights
        state = load_training_weights(weights, map_location="cpu") if isinstance(weights, (str, Path)) else deepcopy(weights)
        state = state.get("model", state)
        is_nemo = not local or _checkpoint_digest(state) == release["state_sha256"]
        config = Classifier.extract_metadata(state)
        getter, _ = resolve_backbone_getter(config["backbone_class"])
        prefix = config["backbone_output_name"] + "."
        complete = any(isinstance(value, torch.Tensor) and not key.startswith(prefix) for key, value in state.items())
        # Legacy frozen-backbone checkpoints contain only the trained head.
        model_args = {} if getter is get_dynamic_model else {"pretrained": not complete}
        self.model, self.preproc = BaseBuilder.build_model(weights=state, device="cpu", dtype=torch.float32, model_args=model_args)
        self.model.to(self.device).eval()
        head = classification_module(self.model)
        self._metadata = deepcopy(head.get_extra_state())
        self._metadata["backend"] = "torch"
        if is_nemo:
            self.preproc = TorchPreprocess(torch, self.device)
            self._metadata.update(model_id="MAMBO_v3", name="Nemo", preprocessing=release["preprocessing"])
        self.resize_size = self._metadata["resize_size"]
        self.reader = make_read_and_resize_fn((self.resize_size, self.resize_size), self.device, torch.uint8)
        if not local and self.preset != "full":
            preset = self.preset
            self._apply_class_mask(release["presets"][preset])
            self.preset = preset
        if class_mask is not None:
            self._apply_class_mask(class_mask)

    @staticmethod
    def _preset_name(value, presets):
        def normalize(text):
            return re.sub(r"[^a-z0-9]", "", text.lower())

        query = normalize(str(value))
        names = ["full", *presets]
        matches = [n for n in names if normalize(n) == query] or [n for n in names if query and normalize(n).startswith(query)]
        if len(matches) != 1:
            raise ValueError(f"Unknown or ambiguous preset: {value!r}")
        return matches[0]

    @property
    def metadata(self):
        return deepcopy(
            {
                **self._metadata,
                "preset": self.preset,
                "precision": self.precision,
                "embedding_dim": self.embedding_dim,
                "input_size": self.input_size,
                "preprocessing": self.preprocessing,
                "cls2idx": self.cls2idx,
            }
        )

    @property
    def input_size(self):
        return self.resize_size

    @property
    def cls2idx(self):
        mapping = self._metadata["cls2idx"]
        if mapping and not isinstance(next(iter(mapping.values())), dict):
            mapping = {"0": mapping}
        return deepcopy({str(k): v for k, v in mapping.items()})

    @property
    def classes(self):
        return [
            [label for label, _ in sorted(rank.items(), key=lambda item: item[1])]
            for _, rank in sorted(self.cls2idx.items(), key=lambda item: int(item[0]))
        ]

    @property
    def class_list(self):
        selected = classification_module(self.model).active_indices
        labels = self.classes[0]
        return labels if selected is None else [labels[i] for i in selected.tolist()]

    @property
    def embedding_dim(self):
        return classification_module(self.model).preclassification_size

    @property
    def preprocessing(self):
        return deepcopy(self._metadata.get("preprocessing", {"resize_size": self.resize_size, "transform": repr(self.preproc)}))

    def load(self, *, embeddings=False):
        return self

    def _apply_class_mask(self, mask):
        if isinstance(mask, int):
            if mask != -1:
                raise ValueError("Only -1 is a valid integer mask flag")
            mask = None
        if mask is not None:
            if hasattr(mask, "detach"):
                mask = mask.detach().cpu().tolist()
            mask = list(mask)
            if mask and all(isinstance(x, (bool, np.bool_)) for x in mask):
                if len(mask) != len(self.classes[0]):
                    raise ValueError("Boolean mask must cover the full vocabulary")
                mask = [i for i, keep in enumerate(mask) if keep]
            mapping = self.cls2idx["0"]
            mask = [mapping[x] if isinstance(x, str) else int(x) for x in mask]
            if not mask or any(i < 0 or i >= len(mapping) for i in mask):
                raise ValueError("Invalid or empty class mask")
            mask = sorted(set(mask))
        set_classification_mask(self.model, mask)
        self.preset = "full" if mask is None else "custom"

    def forward(self, images, *, embeddings=False):
        """Execute already preprocessed NCHW tensors; return device logits and vectors."""
        head = classification_module(self.model)
        with torch.inference_mode():
            with (
                torch.autocast(
                    self.device.type, dtype=torch.bfloat16 if self.precision == "bf16" else torch.float16, enabled=self.precision != "fp32"
                ),
                bypass_submodule(self.model, self.model._backbone_output_name),
            ):
                features = self.model(images.to(device=self.device, dtype=torch.float32))
            with torch.autocast(self.device.type, enabled=False):
                features = features.float()
                return head(features), head.preclassification(features) if embeddings else None

    def _predict(self, x, *, embeddings=False, topk=1, **kwargs):
        if isinstance(x, (str, Path)):
            x = self.reader(str(x))
        elif not isinstance(x, (torch.Tensor, np.ndarray)):
            x = torch.stack([self.reader(str(item)) if isinstance(item, (str, Path)) else torch.as_tensor(item) for item in x])
        x = torch.as_tensor(x)
        if x.ndim == 2:
            x = x.unsqueeze(0)
        if x.ndim == 3:
            x = x.unsqueeze(0)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        raw, vectors = self.forward(self.preproc(x).to(self.device), embeddings=embeddings)
        head = classification_module(self.model)
        result_type = _Prediction if isinstance(raw, list) else Prediction
        result = result_type(raw, topk=topk, **{**head.metadata, **kwargs})
        return (result, vectors) if embeddings else result

    def predict(self, x, **kwargs):
        return self._predict(x, **kwargs)

    def __call__(self, x, **kwargs):
        return self.predict(x, **kwargs)

    def predict_with_embeddings(self, x, **kwargs):
        return self._predict(x, embeddings=True, **kwargs)


def run():
    from mini_trainer.hierarchical.predict import cli
    from mini_trainer.predict import main

    args = cli(model={None: "-M", "type": str, "default": None, "help": "Nemo geographic preset or checkpoint path"})
    model = args.pop("model", None)
    explicit_weights = args["weights"] is not None
    if args["weights"] is None:
        args["weights"] = ensure_weights(model)[0]
    elif model is not None:
        raise ValueError("model and weights are mutually exclusive")
    base = args.get("builder", BaseBuilder)

    class DeploymentBuilder(base):
        @staticmethod
        def build_model(**kwargs):
            model, preproc = base.build_model(**kwargs)
            state = load_training_weights(kwargs["weights"], map_location="cpu")
            if _checkpoint_digest(state.get("model", state)) == _release()["state_sha256"]:
                preproc = TorchPreprocess(torch, kwargs["device"])
            return model, lambda images: preproc(images).float()

    args["builder"] = DeploymentBuilder
    if model is None and (explicit_weights or args.get("class_list") is not None):
        main(**args)
        return
    preset = (
        Predictor._preset_name(model or "europe", _release()["presets"])
        if not model or Path(model).suffix not in (".pt", ".pth")
        else "full"
    )
    if preset == "full":
        main(**args)
    else:
        with TemporaryDirectory(prefix="nemo-preset-") as directory:
            path = Path(directory) / "classes.txt"
            path.write_text("\n".join(_release()["presets"][preset]) + "\n")
            main(**{**args, "class_list": str(path)})


if __name__ == "__main__":
    run()
