import os

import torch
import torchvision.transforms as tt
from torch.utils.data import DataLoader, TensorDataset

from mini_trainer.builders import BaseBuilder
from mini_trainer.logging import configure_loggers
from mini_trainer.train import main


class MockBuilder(BaseBuilder):
    @staticmethod
    def class_spec(*args, **kwargs):
        # Return a dummy spec
        return {"num_classes": 2, "cls2idx": {"class_a": 0, "class_b": 1}}

    @staticmethod
    def build_dataloader(batch_size, device, dtype, **kwargs):
        # Create synthetic data
        # Class 0: mean 0. Class 1: mean 1.
        n = 20
        c, h, w = 3, 5, 5
        data_0 = torch.randn(n, c, h, w)
        data_1 = torch.randn(n, c, h, w) + 1.0
        data = torch.cat([data_0, data_1])
        # labels need to be LongTensor
        labels = torch.cat([torch.zeros(n, dtype=torch.long), torch.ones(n, dtype=torch.long)])

        dataset = TensorDataset(data, labels)

        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        return labels.numpy(), train_loader, val_loader

    @staticmethod
    def build_augmentation(dtype):
        # Must return torchvision.transforms.Compose
        return tt.Compose([])

    @staticmethod
    def build_regularizer(*args, **kwargs):
        # Disable regularization to avoid last_layer_weights dependency on Classifier class
        return lambda x: torch.tensor(0.0)


class TinyMockModel(torch.nn.Module):
    """A tiny mock model for fast unit testing."""

    default_transform = tt.Compose([])

    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 4, kernel_size=3, padding=1),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
        )
        self.fc = torch.nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(self.features(x))


def test_integration_train_cpu(tmp_path):
    # Setup paths
    input_dir = str(tmp_path / "data")
    os.makedirs(input_dir, exist_ok=True)
    # create dummy class dirs so validation passes if it checks
    os.makedirs(os.path.join(input_dir, "class_a"), exist_ok=True)
    os.makedirs(os.path.join(input_dir, "class_b"), exist_ok=True)

    output_dir = str(tmp_path / "output")

    # Run training
    args = {
        "input": input_dir,
        "output": output_dir,
        "epochs": 2,
        "device": "cpu",
        # Use float32 to avoid potential half-precision issues on CPU
        # (though modern torch often handles it, cleaner to use float32 for simple test)
        "dtype": "float32",
        "name": "test_run",
        "builder": MockBuilder,
        "model_builder_kwargs": {"model_type": TinyMockModel(), "pretrained": False},
        # Be verbose to see output if needed
        "logger_builder_kwargs": {"verbose": True, "logger_cls": configure_loggers()},
        # Disable EMA explicitly
        "ema": False,
        "seed": 42,
    }

    main(**args)

    # Check if files were created
    run_dir = os.path.join(output_dir, "test_run")
    assert os.path.exists(run_dir)
    assert os.path.isdir(run_dir)

    # Check weights
    weights_dir = os.path.join(run_dir, "weights")
    assert os.path.exists(weights_dir)
    assert os.path.exists(os.path.join(weights_dir, "last.pt"))
    assert os.path.exists(os.path.join(weights_dir, "checkpoint_last.pth"))

    # Check config
    assert os.path.exists(os.path.join(run_dir, "config.yaml"))

    # Check class spec
    # MockBuilder.class_spec does not write to file, so this file won't exist unless we write it.
    # assert os.path.exists(os.path.join(run_dir, "class_spec.json"))

    # Verify we can execute the model on the data
    # (Checking if training actually did something is harder without asserting loss decrease,
    # but successful execution covers most integration points)

    # Optional: Load best.pt if it exists (it should if validation ran)
    # Note: best.pt is only saved if validation happens.
    # MockBuilder returns val_loader, so validation should run.
    best_weights_path = os.path.join(weights_dir, "best.pt")
    assert os.path.exists(best_weights_path)

    # Test autoloading from a single .pt weights file without passing model_type or other args
    from mini_trainer.modeling import Classifier, classification_module

    loaded_model, loaded_preprocess = Classifier.build(weights=best_weights_path)
    cls_mod = classification_module(loaded_model)
    assert isinstance(cls_mod, Classifier)
    assert cls_mod.metadata["backbone_output_name"] == "fc"
    assert cls_mod.metadata["backbone_class"] == "tests.test_integration_train:TinyMockModel"
    assert loaded_preprocess is not None
    # Test that the custom preprocessing function runs
    dummy_input = torch.randn(3, 5, 5)
    processed = loaded_preprocess(dummy_input)
    assert processed.shape == (3, 5, 5)
