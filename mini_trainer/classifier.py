import os
import warnings
from collections import OrderedDict
from functools import partial
from typing import Any

import torch
import torch.nn as nn
import torchvision
from torchvision.io import ImageReadMode, decode_image

from mini_trainer.utils import make_convert_dtype, recursive_dfs_attr

try:
    from torch.nn.utils.parametrizations import weight_norm
except Exception:  # fallback for older installs
    from torch.nn.utils import weight_norm

_UNSUPPORTED_MODELS = [
    'fasterrcnn_mobilenet_v3_large_320_fpn', 'fasterrcnn_mobilenet_v3_large_fpn', 
    'fasterrcnn_resnet50_fpn', 'fasterrcnn_resnet50_fpn_v2', 
    'fcos_resnet50_fpn', 
    'keypointrcnn_resnet50_fpn', 
    'maskrcnn_resnet50_fpn', 'maskrcnn_resnet50_fpn_v2', 
    'mvit_v1_b', 'mvit_v2_s', 
    'raft_large', 'raft_small', 
    'retinanet_resnet50_fpn', 'retinanet_resnet50_fpn_v2', 
    'ssd300_vgg16', 'ssdlite320_mobilenet_v3_large', 
    'swin3d_b', 'swin3d_s', 'swin3d_t', 'swin_b', 'swin_s', 'swin_t', 'swin_v2_b', 'swin_v2_s', 'swin_v2_t', 
    'vit_b_16', 'vit_b_32', 'vit_h_14', 'vit_l_16', 'vit_l_32'
]


def preprocess(item, transform, func=None):
    """Hook torchvision preprocessing function with load image from file to tensor.
    """
    if isinstance(item, str):
        path = str(item)
        if not os.path.exists(path):
            raise FileNotFoundError("Unable to find image: " + path)
        image = decode_image(path, ImageReadMode.RGB)
    elif isinstance(item, torch.Tensor):
        image = item
    else:
        raise TypeError(f"'item' must be of type `str` or `torch.Tensor`, not {type(item)}")
    image = transform(image)
    if func:
        image = func(image)
    return image
    

def get_model(backbone_model: str | torch.nn.Module, model_args: dict = {},
              classifier_name: str | list[str] = ["classifier", "fc"],
              preprocess_dtype : torch.dtype | None=None):
    """Get torchvision model and preprocessing function by name.
    """
    default_transform = None
    if isinstance(backbone_model, str):
        if backbone_model in _UNSUPPORTED_MODELS:
            raise ValueError(f"The model {backbone_model} is not supported.")
        default_weights = torchvision.models.get_model_weights(backbone_model).DEFAULT
        default_transform = default_weights.transforms(antialias=True)
        backbone_model = torchvision.models.get_model(backbone_model, weights=default_weights, **model_args)
    if not isinstance(backbone_model, nn.Module):
        raise ValueError("backbone_model must be a string or a torch.nn.Module")
    backbone_classifier_name = None
    if isinstance(classifier_name, str):
        classifier_name = [classifier_name]
    for name in classifier_name:
        if hasattr(backbone_model, name):
            backbone_classifier_name = name
            break
    if backbone_classifier_name is None:
        raise AttributeError(f"No classifier found with names {classifier_name}")
    return (
        backbone_model, 
        backbone_classifier_name, 
        partial(
            preprocess, 
            transform=default_transform, 
            func=preprocess_dtype if preprocess_dtype is None else make_convert_dtype(preprocess_dtype)
        )
    )


class Classifier(nn.Module): # noqa: D101 TODO
    _version = 1

    @staticmethod
    @torch.no_grad()
    def _normalize_layer(layer: nn.Linear):
        if layer.bias is not None:
            layer.bias.fill_(-1)
            layer.bias.requires_grad_(False)
        weight_norm(layer, name="weight", dim=0)
        layer.parametrizations.weight.original0.fill_(1.0)
        layer.parametrizations.weight.original0.requires_grad_(False)
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

    def __init__( # noqa: D107
            self, 
            in_features : int, 
            out_features : int, 
            hidden : bool | int=True, 
            droprate : float=0.1,
            normalized : bool=True,
            **metadata
        ):
        super().__init__()
        # Input sanitization and checking
        if not isinstance(in_features, int) or not isinstance(out_features, int):
            raise TypeError(
                f'Supplied classification head input and output dimensions {in_features}x{out_features} '
                f'should be `int`, not `{type(in_features)}`/`{type(out_features)}`.'
            )
        if isinstance(hidden, bool): # Check boolean first because it is a subclass of int
            self.preclassification_size = in_features
        elif isinstance(hidden, int):
            assert hidden > 0
            self.preclassification_size = hidden
        else:
            raise TypeError(f'`hidden` must be an integer or boolean, not ({type(hidden)}): {hidden}.')
        if not isinstance(droprate, (float, int)):
            raise TypeError(f'Dropout-rate `{droprate}` should be a `float` (or `int`), not `{type(droprate)}`.')
        elif not (0 <= droprate <= 1):
            raise ValueError(f'Dropout-rate should be between 0 and 1, not: {droprate}.')
        if not isinstance(normalized, bool):
            raise TypeError(f'Normalized should be a `bool`, not `{normalized}` ({type(normalized)}).')
        
        # Store metadata
        metadata.update({
            "mini_trainer_version" : self._version,
            "in_features" : in_features,
            "out_features" : out_features,
            "hidden" : hidden,
            "droprate" : droprate,
            "normalized" : normalized
        })
        self._metadata = metadata

        # Create one hidden layer     
        self.hidden = hidden and nn.Linear(in_features, self.preclassification_size)

        # Create a dropout layer (if hidden)
        self.dropout = hidden and nn.Dropout(p=droprate)

        # Create a BatchNormalization layer
        self.batch_norm = nn.BatchNorm1d(self.preclassification_size)

        layer = nn.Linear(self.preclassification_size, out_features, bias=True)
        self.linear = self._normalize_layer(layer) if normalized else layer

    def preclassification(self, x : torch.Tensor) -> torch.Tensor:
        if self.hidden:
            x = self.dropout(x)
            x = self.hidden(x)
            x = nn.functional.leaky_relu(x)
        return self.batch_norm(x)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return self.linear(self.preclassification(x))
    
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
                UserWarning
            )

        # Implement migration logic here if it becomes relevant

        self._metadata.update(state)

    @classmethod
    def load(
            cls,
            architecture_class : str,
            architecture_output_name : str,
            architecture : nn.Module,
            state : OrderedDict[str, torch.Tensor | Any] | None,
            device : torch.types.Device,
            dtype : torch.dtype,
            **kwargs
        ):
        """Load weights into model architecture.
        """
        cfg = {
            "backbone_class" : architecture_class,
            "backbone_output_name" : architecture_output_name
        }
        kwargs.update(cfg)
        architecture.add_module(architecture_output_name, cls(**kwargs))
        for k, v in cfg.items():
            setattr(architecture, f'_{k}', v)
        if state is not None:
            try:
                architecture.load_state_dict(state, strict=True)
            except RuntimeError as e:
                if "Missing key(s)" in str(e) and "_extra_state" in str(e):
                    architecture.load_state_dict(state, strict=False)
                    warnings.warn(
                        f'{architecture_class} loaded with `strict=False`, proceed with caution!',
                        UserWarning
                    )
                else:
                    raise e

        architecture.to(device, dtype)        
        return architecture

    @classmethod    
    def build(
            cls,
            model_type : str | None=None, 
            weights : str | OrderedDict[str, torch.Tensor | Any] | None=None, 
            num_classes : list[int] | int | None=None,
            device : torch.types.Device=torch.device("cpu"), 
            dtype : torch.dtype=torch.float32,
            **kwargs
        ):
        if not isinstance(device, torch.device):
            device = torch.device(device)
        cfg = {}
        state = stored_head_name = stored_version = None
        # Parse metadata stored in .pt file if available
        if weights is not None:
            if isinstance(weights, str):
                state = torch.load(
                    f=weights, 
                    map_location=device, 
                    weights_only=True
                )
                state : OrderedDict[str, torch.Tensor | Any] = state.get("model", state)
            else:
                state = weights
            cfg = cls.extract_metadata(state)
            stored_model_type = cfg.pop("backbone_class", None)
            if stored_model_type is None:
                if model_type is None:
                    raise RuntimeError('Unable to infer missing model type from supplied weights.')
            else:
                assert isinstance(stored_model_type, str)
                if stored_model_type != model_type and model_type is not None:
                    warnings.warn(
                        f'Manually specified model type "{model_type}" overridden to "{stored_model_type}"!',
                        UserWarning
                    )
                model_type = stored_model_type
            stored_head_name = cfg.pop("backbone_output_name", None)
            assert stored_head_name is None or isinstance(stored_head_name, str)
            stored_version = cfg.pop("mini_trainer_version")
            assert isinstance(stored_version, int)
        else:
            if model_type is None:
                raise ValueError(
                    f'Building a {cls.__name__} from scratch (w.o. weight file) '
                    f'requires specifying the model type, not {model_type}'
                )
        
        # Build backbone
        architecture, head_name, model_preprocess = get_model(model_type, preprocess_dtype=dtype)
        if not isinstance(architecture, nn.Module):
            raise TypeError(f"Unknown model type `{type(architecture)}`, expected `{nn.Module}`")
        if stored_head_name is not None and stored_head_name != head_name:
            warnings.warn(
                f'Classification head module name "{stored_head_name}" implied in weights '
                f'does not match the derived name "{head_name}"!'
            )

        # Config heuristics
        num_embeddings = recursive_dfs_attr(
            getattr(architecture, head_name), 
            "in_features", 
            lambda x : isinstance(x, int)
        )
        if state is not None:
            for key in list(state.keys()):
                if isinstance(state[key], torch.Tensor):
                    state[key] = state[key].to(device, dtype)
            head_weights = state.get(f"{head_name}.linear.weight", None)
            if head_weights is None:
                head_weights = state.get(f"{head_name}.linear.parametrizations.weight.original1", None)
            if head_weights is not None:
                num_classes, _ = head_weights.shape
            else:
                warnings.warn(
                    'Unable to infer number of classes from supplied weights.',
                    UserWarning
                )
            hidden_layer = state.get(f"{head_name}.hidden.weight", None)
            kwargs.update({"hidden" : isinstance(hidden_layer, torch.Tensor) and hidden_layer.shape[0]})
        if isinstance(num_classes, (list, tuple)):
            num_classes = num_classes[0]
        kwargs.update({
            "in_features" : num_embeddings,
            "out_features" : num_classes
        })

        # Check parity between supplied/heuristic and stored config, and let stored override
        for k, v in cfg.items():
            if kwargs.get(k, None) is None:
                kwargs[k] = v
                continue
            if v != kwargs[k]:
                warnings.warn(
                    f'Model configuration option {k} overriden by value stored in config: ' 
                    f'{kwargs[k]} ==> {v}',
                    UserWarning
                )
                kwargs[k] = v

        # Rebuild (and load) integrated backbone and classifier
        model = cls.load(
            model_type, head_name, architecture, state, device, dtype, 
            **kwargs
        )
        return model, model_preprocess


def last_layer_weights(model : nn.Module):
    """Retrieve the weights of the last layer of a model created with `mini_trainer.classifier.Classifier.build()`.
    """
    backbone_name = getattr(model, "_backbone_output_name", None)
    if backbone_name is None:
        for name, module in model.named_modules():
            if isinstance(module, Classifier):
                setattr(model, "_backbone_output_name", name)
                return module.linear.weight
    else:
        classification_head = getattr(model, backbone_name, None)
        if not isinstance(classification_head, Classifier):
            raise RuntimeError(f"Unexpected classification head type {type(classification_head)} found.")
        return classification_head.linear.weight