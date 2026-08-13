import json
import warnings
from collections import OrderedDict, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, TypeVar

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch._prims_common import DeviceLikeType

from mini_trainer.utils import class_path, cosine_to_zscore, dtype_to_string, import_class, string_to_dtype

from .architectures import get_model
from .prior import prior_from_labels

try:
    from torch.nn.utils.parametrizations import weight_norm
except Exception:  # fallback for older installs
    from torch.nn.utils import weight_norm

from functools import lru_cache

from .context import EmbeddingContext


class Classifier(nn.Module):  # noqa: D101 TODO
    _version = 1

    @classmethod
    @torch.no_grad()
    def init_spherical_repulsion(cls, layer: nn.Module, iterations: int = 100, lr: float = 0.5):
        w = layer.weight
        assert isinstance(w, torch.Tensor)
        nn.init.normal_(w)

        for _ in range(iterations):
            w.div_(w.norm(dim=1, keepdim=True).clamp(min=1e-9))

            # Gradient of sum((w @ w.T)^2) is 4 * (w @ w.T @ w)
            # We subtract the projection onto the vectors to stay on the sphere
            grad = w @ w.t() @ w
            proj = (grad * w).sum(dim=1, keepdim=True) * w
            w.sub_(lr * (grad - proj))

        w.div_(w.norm(dim=1, keepdim=True).clamp(min=1e-9))  # type: ignore
        return layer

    @classmethod
    @torch.no_grad()
    def _normalize_layer(cls, layer: nn.Linear, orthogonal_init: bool = False):
        if layer.bias is not None:
            layer.bias.fill_(0)
            layer.bias.requires_grad_(False)
        if orthogonal_init:
            layer = cls.init_spherical_repulsion(layer)  # type: ignore
        weight_norm(layer, name="weight", dim=0)
        layer.parametrizations.weight.original0.fill_(1)  # type: ignore
        layer.parametrizations.weight.original0.requires_grad_(False)  # type: ignore # Freeze weight row norm
        return layer

    @staticmethod
    def extract_metadata(state: dict[str, Any]) -> dict[str, Any]:
        """Scans the state dictionary for Classifier metadata.
        Returns the config dict if found, otherwise None.
        """
        for key, value in state.items():
            if key.endswith("._extra_state") and isinstance(value, dict):
                if "mini_trainer_version" in value:
                    return value.copy()
        return {}

    @property
    def last_layer_weights(self) -> torch.Tensor | list[torch.Tensor]:
        return self.linear.weight

    def __init__(  # noqa: D107
        self,
        in_features: int,
        out_features: int,
        hidden: bool | int = True,
        droprate: float = 0.1,
        normalized: bool = True,
        active_indices: torch.Tensor | None = None,
        prior: torch.Tensor | list[float] | None = None,
        **metadata,
    ):
        super().__init__()
        # Input sanitization and checking
        if not isinstance(in_features, int) or not isinstance(out_features, int):
            raise TypeError(
                f"Supplied classification head input and output dimensions {in_features}x{out_features} "
                f"should be `int`, not `{type(in_features)}`/`{type(out_features)}`."
            )
        if isinstance(hidden, bool):  # Check boolean first because it is a subclass of int
            self.preclassification_size = in_features
        elif isinstance(hidden, int):
            assert hidden > 0
            self.preclassification_size = hidden
        else:
            raise TypeError(f"`hidden` must be an integer or boolean, not ({type(hidden)}): {hidden}.")
        if not isinstance(droprate, (float, int)):
            raise TypeError(f"Dropout-rate `{droprate}` should be a `float` (or `int`), not `{type(droprate)}`.")
        elif not (0 <= droprate <= 1):
            raise ValueError(f"Dropout-rate should be between 0 and 1, not: {droprate}.")
        if not isinstance(normalized, bool):
            raise TypeError(f"Normalized should be a `bool`, not `{normalized}` ({type(normalized)}).")

        # Store metadata
        metadata.update(
            {
                "mini_trainer_version": self._version,
                "classifier_class": class_path(self),
                "in_features": in_features,
                "out_features": out_features,
                "hidden": hidden,
                "droprate": droprate,
                "normalized": normalized,
                "prior": prior.tolist() if isinstance(prior, torch.Tensor) else prior,
            }
        )
        self._metadata = metadata

        # Create one hidden layer
        self.hidden = hidden and nn.Linear(in_features, self.preclassification_size)

        # Create a dropout layer (if hidden)
        self.dropout = hidden and nn.Dropout(p=droprate)

        # Create a BatchNormalization layer
        self.batch_norm = nn.BatchNorm1d(self.preclassification_size)

        # Create linear classification layer
        layer = nn.Linear(self.preclassification_size, out_features, bias=True)
        self.normalized = normalized
        self.linear = self._normalize_layer(layer, True) if self.normalized else layer
        if self._metadata.get("prior", None) is not None:
            self.linear.bias.data[:] = torch.tensor(
                data=self._metadata["prior"], device=self.linear.weight.device, dtype=self.linear.weight.dtype
            )

        # Prepare class masking buffer
        if active_indices is not None:
            self.register_buffer("active_indices", active_indices, persistent=True)
        else:
            self.register_buffer("active_indices", None)
        self._dirty_cache = defaultdict(lambda: True)
        self.register_buffer("_linear_weight", torch.empty(0), persistent=False)
        self.register_buffer("_linear_bias", torch.empty(0), persistent=False)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs) -> None:
        retval = super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        self._dirty_cache.clear()
        return retval

    def set_active_features(self, indices: list[int] | torch.Tensor | np.ndarray | None = None):
        """Mask a selection of output features (classes).

        Args:
            indices: Indices to (reversibly) mask in forward pass. If None the mask is disabled.
        """
        self._dirty_cache.clear()
        if indices is None:
            self.active_indices = None
            return
        device = next(self.parameters()).device
        self.active_indices = torch.as_tensor(indices, dtype=torch.long, device=device).clone()
        self.active_indices = self.active_indices.sort().values
        _ = self._preprocess_metadata()

    def _weight_bias(self, i: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        if self._dirty_cache["_weight_bias"] or self.training:
            weight, bias = self.linear.weight, self.linear.bias
            if self.active_indices is not None:
                weight = weight.index_select(0, self.active_indices)
                if bias is not None:
                    bias = bias.index_select(0, self.active_indices)
            if self.training:
                self._dirty_cache["_weight_bias"] = True
                return weight, bias
            self._linear_weight = weight.detach()
            self._linear_bias = bias.detach() if bias is not None else bias
            self._dirty_cache["_weight_bias"] = False
        return self._linear_weight, self._linear_bias

    def _reshape_backbone_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        D = self.preclassification_size

        if x.ndim == 2:
            if x.shape[1] == D:
                return x
            if x.shape[1] == 1 and x.numel() % D == 0:
                return x.reshape(-1, D)
            raise ValueError(f"Unexpected 2D preclassification input shape: {tuple(x.shape)}")

        if x.ndim == 3:
            if x.shape[-1] == D:
                if x.shape[1] == 1:
                    return x[:, 0, :]
                raise ValueError(f"Ambiguous 3D preclassification input shape {tuple(x.shape)}; expected singleton token dimension.")
            if x.shape[1] == D and x.shape[-1] == 1:
                return x[:, :, 0]
            raise ValueError(f"Unexpected 3D preclassification input shape: {tuple(x.shape)}")

        if x.ndim == 4:
            if x.shape[1] == D and x.shape[-2:] == (1, 1):
                return x.flatten(1)
            if x.shape[-1] == D and x.shape[1:3] == (1, 1):
                return x.squeeze(1).squeeze(1)
            raise ValueError(f"Unexpected 4D preclassification input shape: {tuple(x.shape)}")

        raise ValueError(f"Unexpected preclassification input shape: {tuple(x.shape)}")

    def preclassification(self, x: torch.Tensor) -> torch.Tensor:
        x = self._reshape_backbone_embeddings(x)
        if self.hidden:
            x = self.dropout(x)  # type: ignore
            x = self.hidden(x)  # type: ignore
            x = F.leaky_relu(x)
        if self.normalized:
            # Output is unit vectors
            return F.normalize(x, 2, -1)
        else:
            # Output is MVN
            return self.batch_norm(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight, bias = self._weight_bias()
        embeddings = self.preclassification(x)
        if EmbeddingContext.active():
            EmbeddingContext.set(embeddings)
        if self.normalized:
            return cosine_to_zscore(F.linear(embeddings, weight=weight), self.preclassification_size) + bias
        else:
            return F.linear(embeddings, weight=weight, bias=bias)

    def get_extra_state(self):
        return self._metadata

    def set_extra_state(self, state: Any):
        if state is None:
            return

        loaded_version = state.get("mini_trainer_version", 0)
        if loaded_version != self._version:
            warnings.warn(
                f"Version mismatch: Loading Classifier weights saved with mini_trainer_version {loaded_version} "
                f"into current code version {self._version}. This may result in unexpected behavior.",
                UserWarning,
            )

        # Implement migration logic here if it becomes relevant

        self._metadata.update(state)

    def _preprocess_metadata(self, cls2idx=None, **kwargs):
        if self._dirty_cache["_preprocess_metadata"] or cls2idx is not None:
            metadata = self._metadata.copy()
            if cls2idx is None:
                cls2idx = metadata.get("cls2idx", None)
            active_indices = self.active_indices
            if active_indices is not None and cls2idx is not None:
                if isinstance(active_indices, torch.Tensor):
                    active_indices = active_indices.tolist()
                active_indices = sorted(active_indices)
                reindex = {old: new for new, old in enumerate(active_indices)}
                cls2idx = {k: reindex[v] for k, v in cls2idx.items() if v in reindex}
                metadata["cls2idx"] = cls2idx
            self._preprocessed_metadata = metadata
            self._dirty_cache["_preprocess_metadata"] = False
        return {**self._preprocessed_metadata, **kwargs}

    @property
    def metadata(self):
        return self._preprocess_metadata()

    @torch.no_grad()
    def predict(self, x, topk: int = 1, **kwargs):
        return Prediction(self(x), topk=topk, **self._preprocess_metadata(**kwargs))

    @classmethod
    def load(
        cls,
        architecture_class: str,
        architecture_output_name: str,
        architecture: nn.Module,
        state: OrderedDict[str, torch.Tensor | Any] | None,
        device: torch.device,
        dtype: torch.dtype,
        strict: bool = True,
        train_labels: list[list[int]] | list[int] | None = None,
        cls2idx: dict[str, int] | None = None,
        **kwargs,
    ):
        """Load weights into model architecture."""
        if state is not None:
            indices_key = f"{architecture_output_name}.active_indices"
            if indices_key in state:
                kwargs["active_indices"] = state[indices_key].long()
        cfg = {"backbone_class": architecture_class, "backbone_output_name": architecture_output_name}
        kwargs.update(cfg)
        if cls2idx is not None:
            kwargs["cls2idx"] = cls2idx
            if train_labels is not None:
                kwargs["prior"] = prior_from_labels(train_labels, cls2idx=cls2idx)
        with device:
            architecture.add_module(architecture_output_name, cls(**kwargs))
        for k, v in cfg.items():
            setattr(architecture, f"_{k}", v)
        if state is not None:
            try:
                load_result = architecture.load_state_dict(state, strict=strict)
            except RuntimeError as e:
                if "Missing key(s)" in str(e) and "_extra_state" in str(e):
                    load_result = architecture.load_state_dict(state, strict=False)
                    warnings.warn(f"{architecture_class} loaded with `strict=False`, proceed with caution!\n{e}", UserWarning)
                else:
                    raise
            if load_result.missing_keys:
                warnings.warn(f"Missing keys when loading {architecture_class}[{cls.__name__}]: {load_result.missing_keys}")

        architecture.to(device)
        return architecture

    @classmethod
    def build(
        cls,
        model_type: str | nn.Module | None = None,
        weights: str | OrderedDict[str, torch.Tensor | Any] | None = None,
        num_classes: list[int] | int | None = None,
        device: DeviceLikeType = torch.device("cpu"),
        dtype: torch.dtype | None = None,
        preprocess_dtype: torch.dtype | None = None,
        transform: Any = None,
        model_args: dict = {},
        **kwargs,
    ):
        if not isinstance(device, torch.device):
            device = torch.device(device)
        cfg = {}
        state: None | OrderedDict[str, torch.Tensor | Any]
        state = stored_head_name = stored_version = None
        # Parse metadata stored in .pt file if available
        if weights is not None:
            if isinstance(weights, str):
                state = torch.load(f=weights, map_location=device, weights_only=True)
                state = state.get("model", state)  # type: ignore
            else:
                state = weights
            if state is None:
                raise RuntimeError(f"Failed to extract state from {weights=}")
            cfg = cls.extract_metadata(state)
            stored_dtype = cfg.pop("_dtype", None)
            if dtype is None:
                dtype = string_to_dtype(stored_dtype)
            stored_preprocess_dtype = cfg.pop("preprocess_dtype", None)
            if preprocess_dtype is None and stored_preprocess_dtype is not None:
                preprocess_dtype = string_to_dtype(stored_preprocess_dtype)
            stored_model_type = cfg.pop("backbone_class", None)
            stored_classifier_type = cfg.pop("classifier_class", None)
            if stored_classifier_type is not None:
                stored_classifier_type = import_class(stored_classifier_type)
                if stored_classifier_type is not cls:
                    cls = stored_classifier_type
                    assert issubclass(cls, Classifier)
            if stored_model_type is None:
                if model_type is None:
                    raise RuntimeError("Unable to infer missing model type from supplied weights.")
            else:
                assert isinstance(stored_model_type, str)
                if model_type is None:
                    model_type = stored_model_type
                elif isinstance(model_type, str):
                    if stored_model_type != model_type:
                        warnings.warn(
                            f'Manually specified model type "{model_type}" overridden to "{stored_model_type}"!',
                            UserWarning,
                        )
                        model_type = stored_model_type
                else:
                    model_type_str = class_path(model_type)
                    if stored_model_type != model_type_str:
                        warnings.warn(
                            f'Manually specified model type "{model_type_str}" does not match stored model type "{stored_model_type}"!',
                            UserWarning,
                        )
            stored_head_name = cfg.pop("backbone_output_name", None)
            assert stored_head_name is None or isinstance(stored_head_name, str)
            stored_version = cfg.pop("mini_trainer_version", None)
            assert stored_version is None or isinstance(stored_version, int)
        else:
            if model_type is None:
                raise ValueError(
                    f"Building a {cls.__name__} from scratch (w.o. weight file) requires specifying the model type, not {model_type}"
                )

        # Build backbone
        if dtype is None:
            dtype = torch.float32
        preprocess_dtype = preprocess_dtype or dtype

        resize_size = kwargs.get("resize_size") or cfg.get("resize_size")
        model_args = {**model_args, "resize_size": resize_size} if resize_size else model_args

        architecture, head_name, model_preprocess, num_embeddings, preferred_size = get_model(
            model_type, model_args=model_args, preprocess_dtype=preprocess_dtype, transform=transform
        )
        if not isinstance(architecture, nn.Module):
            raise TypeError(f"Unknown model type `{type(architecture)}`, expected `{nn.Module}`")

        resize_size = resize_size or preferred_size

        if stored_head_name is not None and stored_head_name != head_name:
            warnings.warn(
                f'Classification head module name "{stored_head_name}" implied in weights does not match the derived name "{head_name}"!',
                UserWarning,
            )

        if state is not None:
            for key in list(state.keys()):
                if isinstance(state[key], torch.Tensor):
                    if key.startswith(head_name):
                        state[key] = state[key].to(device, dtype)
                    else:
                        state[key] = state[key].to(device)
            head_weights = state.get(f"{head_name}.linear.weight", None)
            if head_weights is None:
                head_weights = state.get(f"{head_name}.linear.parametrizations.weight.original1", None)
            if head_weights is not None:
                num_classes, _ = head_weights.shape
            else:
                warnings.warn("Unable to infer number of classes from supplied weights.", UserWarning)
            hidden_layer = state.get(f"{head_name}.hidden.weight", None)
            kwargs.update({"hidden": isinstance(hidden_layer, torch.Tensor) and hidden_layer.shape[0]})
        if isinstance(num_classes, (list, tuple)):
            num_classes = num_classes[0]
        kwargs.update(
            {
                "in_features": num_embeddings,
                "out_features": num_classes,
                "_dtype": dtype_to_string(dtype),
                "preprocess_dtype": dtype_to_string(preprocess_dtype),
                "preferred_size": preferred_size,
                "resize_size": resize_size,
            }
        )

        # Check parity between supplied/heuristic and stored config, and let stored override
        for k, v in cfg.items():
            if kwargs.get(k, None) is None:
                kwargs[k] = v
                continue
            if v != kwargs[k]:
                warnings.warn(f"Model configuration option '{k}' overriden by value stored in config: {kwargs[k]} ==> {v}", UserWarning)
                kwargs[k] = v
        # Rebuild (and load) integrated backbone and classifier
        model_type_str = model_type if isinstance(model_type, str) else class_path(model_type)
        model = cls.load(model_type_str, head_name, architecture, state, device, dtype, **kwargs)
        return model, model_preprocess


@dataclass
class PredictionItem:
    """Simple data container.
    Auto-converts inputs to native Python types on initialization.
    """

    label: str
    confidence: float
    index: int

    def __post_init__(self):
        # Factory coercion: ensures native types immediately
        self.label = str(self.label)
        self.confidence = float(self.confidence)
        self.index = int(self.index)

    def __repr__(self):
        lbl = f"'{self.label}'" if self.label else f"Index {self.index}"
        return f"{lbl}: {self.confidence:.2%}"

    def to_dict(self):
        return {"label": self.label, "confidence": self.confidence, "index": self.index}


T = TypeVar("T", bound=PredictionItem)
I = TypeVar("I")  # noqa: E741


class BasePrediction[T: PredictionItem, I]:
    """Standard PyTorch Prediction class.
    Assumes inputs are Tensors. Subclass to handle other types.
    """

    ITEM_CLASS: type[T] = PredictionItem  # type: ignore
    items: list[T]

    def __init__(  # noqa: D107
        self, prediction: I, topk: int = 1, cls2idx: dict[str, int] | None = None, **kwargs
    ):
        self.topk = topk
        self.cls2idx = cls2idx
        self.metadata = kwargs

        self.logits, self.indices = self._process(prediction)
        self.labels = self._translate()
        self.confidence = self._extract_confidence(prediction)
        if self.topk == 1:
            self.items = [
                self.ITEM_CLASS(label=lab[0], confidence=conf[0], index=idx[0])
                for lab, conf, idx in zip(self.labels, self.confidence, self.indices)
            ]
        else:
            warnings.warn(f"{topk=}, all values except topk=1 are experimental and will result in unexpected behaviour.", UserWarning)
            self.items = [  # type: ignore
                [self.ITEM_CLASS(label=labi, confidence=confi, index=idxi) for labi, confi, idxi in zip(lab, conf, idx)]
                for lab, conf, idx in zip(self.labels, self.confidence, self.indices)
            ]

    # --- Standard Implementations (override) ---
    @property
    def idx2cls(self):
        raise NotImplementedError()

    def _process(self, raw_prediction: I):
        raise NotImplementedError()

    def _translate(self):
        raise NotImplementedError()

    def _extract_confidence(self, raw_prediction: I):
        raise NotImplementedError()

    # --- Convenience & Serialization (do not override) ---
    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        return self.items[idx]

    def __iter__(self):
        yield from self.items

    def __repr__(self) -> str:
        if not self:
            return f"<{type(self).__name__}(Empty)>"
        return f"<{type(self).__name__}(topk={self.topk})>\n" + "\n".join([f"\t{item!r}" for item in self])

    def to_dict(self) -> list[dict]:
        return [item.to_dict() for item in self]

    def save(self, path: str):
        data = {
            "results": self.to_dict(),
            "metadata": self.metadata,
            "config": {"topk": self.topk, "cls2idx": self.cls2idx if isinstance(self.cls2idx, dict) else str(type(self.cls2idx))},
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)


class Prediction(BasePrediction[PredictionItem, torch.Tensor]):  # noqa: D101
    @property
    @lru_cache(1)
    def idx2cls(self):
        if self.cls2idx:
            return {v: k for k, v in self.cls2idx.items()}
        return None

    def _process(self, raw_prediction):
        dim = raw_prediction.shape[-1]
        k = self.topk if 1 <= self.topk < dim else dim
        return torch.topk(raw_prediction, k)

    def _translate(self):
        if self.idx2cls:
            _idx2cls = self.idx2cls.copy()

            def fmt_idx(i: int | torch.Tensor):
                if isinstance(i, torch.Tensor):
                    i = int(i.item())
                return _idx2cls[i]
        else:

            def fmt_idx(i: int | torch.Tensor):
                if isinstance(i, torch.Tensor):
                    i = int(i.item())
                return str(i)

        return [[fmt_idx(i) for i in idxs] for idxs in self.indices]

    def _extract_confidence(self, raw_prediction):
        if not (
            torch.all(raw_prediction >= 0)
            and torch.isclose(
                raw_prediction.sum(),
                torch.tensor(1.0, dtype=raw_prediction.dtype),
                atol=1e-3 if raw_prediction.dtype == torch.float16 else 1e-5,
            )
        ):
            raw_prediction = raw_prediction.softmax(dim=-1)

        return raw_prediction.gather(-1, self.indices)


@contextmanager
def bypass_submodule(model: nn.Module, submodule_path: str):
    """Temporarily replaces a submodule with nn.Identity() for a forward pass.

    Args:
        model: The parent PyTorch module.
        submodule_path: The dotted path to the submodule (e.g., 'backbone.layer3.conv1').
    """
    # 1. Traverse the dotted path to find the direct parent of the target
    parts = submodule_path.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)

    target_name = parts[-1]

    # 2. Keep a reference to the original submodule
    original_module = getattr(parent, target_name)

    try:
        # 3. Swap in the no-op module
        setattr(parent, target_name, nn.Identity())
        yield

    finally:
        # 4. Guarantee restoration, even if an error is thrown during the yield
        setattr(parent, target_name, original_module)


def predict(model: nn.Module, x: torch.Tensor, topk: int = 1, **kwargs):
    cls_mod = classification_module(model)
    with bypass_submodule(model, model._backbone_output_name):  # type: ignore
        return cls_mod.predict(model(x), topk=topk, **kwargs)


def backbone(model: nn.Module):
    """This method is suitable for accessing the backbone parameters/modules, but not for inference or forward pass.

    If you wish to omit the forward pass of the classification module, use `bypass_submodule` instead (see `predict`).
    """
    model = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model
    head_name = getattr(model, "_backbone_output_name", None)
    if head_name is None:
        try:
            classification_module(model)
            head_name = getattr(model, "_backbone_output_name", None)
        except Exception:
            pass

    children = []
    for name, child in model.named_children():
        if head_name and name == head_name:
            continue
        children.append(child)

    if head_name is None or len(children) == len(list(model.children())):
        children = list(model.children())[:-1]

    return nn.Sequential(*children, nn.Flatten(start_dim=1, end_dim=-1))


def classification_module(model: nn.Module):
    """Retrieve the classification module of a model created with `mini_trainer.classifier.Classifier.build()`."""
    # Unwrap torch.compile (OptimizedModule) and DDP wrappers
    if hasattr(model, "_orig_mod"):
        _orig = model._orig_mod
        assert isinstance(_orig, nn.Module)
        model = _orig
    model = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model
    backbone_name = getattr(model, "_backbone_output_name", None)
    if backbone_name is None:
        for name, module in model.named_modules():
            if isinstance(module, Classifier):
                setattr(model, "_backbone_output_name", name)
                return module
        raise RuntimeError("Unable to find classification module.")
    else:
        module = getattr(model, backbone_name, None)
        if not isinstance(module, Classifier):
            raise RuntimeError(f"Unexpected classification head type {type(module)} found.")
        return module
