"""Exercise an installed wheel from a disposable directory with only core dependencies."""

import csv
import importlib
import importlib.metadata
import importlib.resources
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import torch
import torchvision.transforms.v2 as transforms
from PIL import Image

import mini_trainer
from mini_trainer.builders import BaseBuilder
from mini_trainer.modeling import Classifier
from mini_trainer.predict import main as predict
from mini_trainer.train import main as train


class TinyBackbone(torch.nn.Module):
    default_transform = transforms.Compose([transforms.ToDtype(torch.float32, scale=True)])

    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten())
        self.fc = torch.nn.Linear(3, 2)

    def forward(self, image):
        return self.fc(self.features(image))


class SmokeBuilder(BaseBuilder):
    @staticmethod
    def build_augmentation(**kwargs):
        return transforms.Compose([torch.nn.Identity()])


def main():
    torch.set_num_threads(1)
    assert Path(mini_trainer.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
    for optional in ("tensorboard", "wandb", "pyarrow", "scipy", "timm", "transformers", "open_clip"):
        assert importlib.util.find_spec(optional) is None, f"Optional package leaked into minimal environment: {optional}"
    for module in ("data", "modeling", "training", "logging", "train", "predict", "hierarchical.train", "hierarchical.predict"):
        importlib.import_module(f"mini_trainer.{module}")
    blacklist = importlib.resources.files("mini_trainer.modeling.architectures").joinpath("blacklist.json")
    assert isinstance(json.loads(blacklist.read_text()), dict)
    for entry in importlib.metadata.distribution("mini_trainer").entry_points:
        if entry.group == "console_scripts":
            subprocess.run([str(Path(sys.executable).parent / entry.name), "--help"], check=True, timeout=60, capture_output=True)

    assets = importlib.resources.files("mini_trainer.visualization.prototype_space")
    for name in ("report.html", "launcher.html", "photos.js", "projection.js", "thumbnails.js", "state.js"):
        assert assets.joinpath(name).read_text(), f"Missing packaged explorer asset: {name}"
    missing_explorer = subprocess.run(
        [str(Path(sys.executable).parent / "mt_explore"), "--no-browser"], capture_output=True, text=True, timeout=30
    )
    assert missing_explorer.returncode != 0 and "mini_trainer[explorer]" in missing_explorer.stderr

    root = Path.cwd()
    data = root / "images"
    for label, color in (("dark", (20, 30, 40)), ("light", (200, 210, 220))):
        (data / label).mkdir(parents=True)
        for index in range(6):
            Image.new("RGB", (8, 8), color).save(data / label / f"{index}.png")
    train(
        input=str(data),
        output=str(root / "runs"),
        name="smoke",
        epochs=1,
        size=8,
        device="cpu",
        dtype="float32",
        seed=42,
        builder=SmokeBuilder,
        model_builder_kwargs={"model_type": TinyBackbone(), "hidden": False, "normalized": False},
        dataloader_builder_kwargs={"batch_size": 2, "num_workers": 0, "cache": "none", "train_proportion": 0.5},
        lr_schedule_builder_kwargs={"warmup_epochs": 0},
        regularizer_builder_kwargs={"strength": 0},
    )
    weights = str(root / "runs" / "smoke" / "weights" / "last.pt")
    model, preprocess = Classifier.build(weights=weights)
    model.eval()
    with torch.inference_mode():
        scores = model(preprocess(torch.zeros(2, 3, 8, 8, dtype=torch.uint8)))
    assert scores.shape == (2, 2) and torch.isfinite(scores).all()
    predict(
        input=str(data),
        weights=weights,
        output=str(root / "predictions"),
        name="smoke",
        device="cpu",
        dtype="float32",
        dataloader_builder_kwargs={"batch_size": 2, "num_workers": 0},
        collector_cls_kwargs={"scientific_names": False},
    )
    csv_files = list((root / "predictions").rglob("*.csv"))
    assert csv_files, "Prediction did not produce a CSV artifact"
    row_counts = []
    for path in csv_files:
        with path.open() as handle:
            row_counts.append(len(list(csv.DictReader(handle))))
    assert 12 in row_counts, f"Expected predictions for all 12 images, found row counts {row_counts}"
    print("Installed wheel: minimal imports, CLI help, resources, training, reload, and prediction passed")


if __name__ == "__main__":
    main()
