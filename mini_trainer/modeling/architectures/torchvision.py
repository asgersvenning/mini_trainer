from collections.abc import Callable

import torchvision

_UNSUPPORTED_MODELS = [
    "squeezenet1_0",
    "squeezenet1_1",
    "fasterrcnn_mobilenet_v3_large_320_fpn",
    "fasterrcnn_mobilenet_v3_large_fpn",
    "fasterrcnn_resnet50_fpn",
    "fasterrcnn_resnet50_fpn_v2",
    "fcos_resnet50_fpn",
    "keypointrcnn_resnet50_fpn",
    "maskrcnn_resnet50_fpn",
    "maskrcnn_resnet50_fpn_v2",
    "mvit_v1_b",
    "mvit_v2_s",
    "raft_large",
    "raft_small",
    "retinanet_resnet50_fpn",
    "retinanet_resnet50_fpn_v2",
    "ssd300_vgg16",
    "ssdlite320_mobilenet_v3_large",
    "swin3d_b",
    "swin3d_s",
    "swin3d_t",
]


def get_torchvision_model(model: str, default_transform: Callable | None = None, **kwargs):
    if model in _UNSUPPORTED_MODELS:
        raise ValueError(f"The model {model} is not supported.")
    default_weights = torchvision.models.get_model_weights(model).DEFAULT
    if default_transform is None:
        try:
            default_transform = default_weights.transforms(antialias=True)
        except TypeError as e:
            if "unexpected keyword argument 'antialias'" not in str(e):
                raise
            default_transform = default_weights.transforms()
    return torchvision.models.get_model(model, weights=default_weights, **kwargs), default_transform
