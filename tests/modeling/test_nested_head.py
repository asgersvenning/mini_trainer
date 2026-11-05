from collections import OrderedDict

import pytest
import torch
import torchvision.transforms as tt
from torch import nn

from mini_trainer.builders import BaseBuilder
from mini_trainer.modeling import Classifier, classification_module
from mini_trainer.modeling.classifier import backbone, predict


class NestedHeadModel(nn.Module):
    """A timm-style backbone whose classifier is nested (``head.fc``), as in timm's ConvNeXt."""

    default_transform = tt.Compose([])

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 4, kernel_size=3, padding=1), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Sequential(OrderedDict(norm=nn.LayerNorm(4), fc=nn.Linear(4, 2)))

    def get_classifier(self):
        return self.head.fc

    def forward(self, x):
        return self.head(self.features(x))


def test_nested_classifier_is_replaced_in_place():
    model, _ = Classifier.build(model_type=NestedHeadModel(), num_classes=3, model_args={"pretrained": False}, hidden=True)
    assert model._backbone_output_name == "head.fc"
    head = classification_module(model)
    assert model.head.fc is head and isinstance(model.head.norm, nn.LayerNorm)  # the backbone's own norm stays
    x = torch.rand(2, 3, 8, 8)
    assert model(x).shape == (2, 3)
    assert predict(model, x) is not None
    rates = dict(head_lr=1e-3, backbone_lr=1e-4, head_weight_decay=0.01, backbone_weight_decay=0.01)
    groups = BaseBuilder.parameter_groups(model, **rates)
    trainable = {id(p) for p in head.parameters() if p.requires_grad}
    assert {id(p) for g in groups if g["name"].startswith("head") for p in g["params"]} == trainable
    with pytest.raises(NotImplementedError):
        backbone(model)
