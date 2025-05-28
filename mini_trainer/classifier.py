import os
from collections import OrderedDict
from functools import partial
from typing import Any, Optional, Union

import torch
import torch.nn as nn
import torchvision
from torchvision.io import ImageReadMode, decode_image

from mini_trainer.utils import convert2fp16

_UNSUPPORTED_MODELS = [
    'fasterrcnn_mobilenet_v3_large_320_fpn', 'fasterrcnn_mobilenet_v3_large_fpn', 'fasterrcnn_resnet50_fpn', 'fasterrcnn_resnet50_fpn_v2', 
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

def preprocess(item, transform, func):
    if isinstance(item, str):
        path = str(item)
        if not os.path.exists(path):
            raise FileNotFoundError("Unable to find image: " + path)
        image = decode_image(path, ImageReadMode.RGB)
    elif isinstance(item, torch.Tensor):
        image = item
    else:
        raise TypeError(f"'item' must be of type `str` or `torch.Tensor`, not {type(item)}")
    return transform(func(image))

def get_model(backbone_model: Union[str, torch.nn.Module], model_args: dict = {},
              classifier_name: Union[str, list[str]] = ["classifier", "fc"]):
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

    return backbone_model, backbone_classifier_name, partial(preprocess, transform=default_transform, func=convert2fp16)

class Classifier(nn.Module):
    def __init__(self, in_features : int, out_features : int, hidden : bool=False):
        super().__init__()
        # Create a BatchNormalization Layer
        self.batch_norm = nn.BatchNorm1d(in_features)

        # Create one hidden layer
        self.hidden = hidden and nn.Linear(in_features, in_features)

        # Create a standard linear layer.
        self.linear = nn.Linear(in_features, out_features, bias=True)
        
        # Set the bias to -1 and freeze it.
        with torch.no_grad():
            self.linear.bias.fill_(-1)
        self.linear.bias.requires_grad_(False)

    def forward(self, x):
        if self.hidden:
            x = nn.functional.leaky_relu(self.hidden(x), True)
        x = self.batch_norm(x)
        return self.linear(x)

    @classmethod
    def load(
        cls,
        architecture_class : str,
        architecture_output_name : str,
        architecture : nn.Module,
        state : Optional[OrderedDict[str, Union[torch.Tensor, Any]]],
        device : torch.types.Device,
        dtype : torch.dtype,
        **kwargs
    ):
        """
        Load weights into model architecture
        """
        architecture.add_module(architecture_output_name, cls(**kwargs))
        setattr(architecture, "_backbone_class", architecture_class)
        setattr(architecture, "_backbone_output_name", architecture_output_name)
        if state is not None:
            architecture.load_state_dict(state)
        architecture.to(device, dtype)
        
        return architecture

    @classmethod    
    def build(
        cls,
        model_type : str, 
        weights : Optional[Union[str, OrderedDict[str, Union[torch.Tensor, Any]]]]=None, 
        num_classes : Optional[Union[list[int], int]]=None,
        device=torch.device("cpu"), 
        dtype=torch.float32,
        **kwargs
    ):
        architecture, head_name, model_preprocess = get_model(model_type)
        if not isinstance(architecture, nn.Module):
            raise TypeError(f"Unknown model type `{type(architecture)}`, expected `{nn.Module}`")
        
        num_embeddings = getattr(architecture, head_name)[1].in_features
        state = None

        if weights is not None:
            if isinstance(weights, str):
                state : OrderedDict[str, torch.Tensor] = torch.load(weights, device, weights_only=True)
            else:
                state = weights
            for key in list(state.keys()):
                if isinstance(state[key], torch.Tensor):
                    state[key] = state[key].to(device, dtype)
            num_classes, _ = state[f"{head_name}.linear.weight"].shape
        else:
            if isinstance(num_classes, list):
                # Here we assume that the number of classes for each level has been passed 
                # and that the number of classes at the leaf level is contained in the first element
                num_classes = num_classes[0]
            if not isinstance(num_classes, int):
                raise RuntimeError('Unable to build classifier with unknown number of output classes. If `weights` is not passed (`None`), the number of classes, `num_classes`, must be specified.')
        
        return cls.load(model_type, head_name, architecture, state, device, dtype, in_features=num_embeddings, out_features=num_classes, **kwargs), model_preprocess


def last_layer_weights(model : nn.Module):
    classification_head = getattr(model, getattr(model, "_backbone_output_name", None), None)
    if classification_head is None:
        return None
    elif not isinstance(classification_head, Classifier):
        raise RuntimeError(f"Unexpected classification head type {type(classification_head)} found.")
    return classification_head.linear._parameters["weight"].data