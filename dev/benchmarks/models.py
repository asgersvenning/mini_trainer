"""Offline backbones for fast, reproducible training benchmarks."""

import torch
from torchvision.transforms.v2 import Compose

from mini_trainer.builders import BaseBuilder


class ColorMean(torch.nn.Module):
    """Linear classifier on channel means; sufficient for the synthetic oracle task."""

    default_transform = Compose([torch.nn.Identity()])

    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten())
        self.fc = torch.nn.Linear(3, 4)

    def forward(self, images):
        return self.fc(self.features(images))


class NoAugmentationBuilder(BaseBuilder):
    @staticmethod
    def build_augmentation(dtype):
        return Compose([torch.nn.Identity()])


class TinyConv(torch.nn.Module):
    """Small spatial backbone for MNIST and real-image pipeline checks."""

    default_transform = Compose([torch.nn.Identity()])

    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(16, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((4, 4)),
            torch.nn.Flatten(),
        )
        self.fc = torch.nn.Linear(512, 10)

    def forward(self, images):
        return self.fc(self.features(images))


class DenseImageMLP(torch.nn.Module):
    """Spatial image MLP for measuring QT when Linear dominates the backbone.

    This intentionally dense architecture is a compute profile, not a proposed
    replacement for CNNs or a model-quality baseline. Its fixed input pooling
    keeps construction and checkpoint restoration independent of dataset size.
    """

    default_transform = Compose([torch.nn.Identity()])

    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d((28, 28)),
            torch.nn.Flatten(),
            torch.nn.Linear(3 * 28 * 28, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, 2048),
            torch.nn.ReLU(),
        )
        self.fc = torch.nn.Linear(2048, 10)

    def forward(self, images):
        return self.fc(self.features(images))
