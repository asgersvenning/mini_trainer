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


def test_integration_resume_train(tmp_path):
    from mini_trainer.config import load_yaml_config

    input_dir = str(tmp_path / "data")
    os.makedirs(os.path.join(input_dir, "class_a"), exist_ok=True)
    os.makedirs(os.path.join(input_dir, "class_b"), exist_ok=True)

    output_dir = str(tmp_path / "output")
    run_name = "resume_test_run"
    run_dir = os.path.join(output_dir, run_name)

    # 1. Fresh run with resume=True when no checkpoint exists yet
    config_fresh = load_yaml_config(path=None, resume=True, output_dir=run_dir)
    assert "checkpoint" not in config_fresh

    args = {
        "input": input_dir,
        "output": output_dir,
        "epochs": 1,
        "device": "cpu",
        "dtype": "float32",
        "name": run_name,
        "builder": MockBuilder,
        "model_builder_kwargs": {"model_type": TinyMockModel(), "pretrained": False},
        "logger_builder_kwargs": {"verbose": False, "logger_cls": configure_loggers()},
        "ema": False,
        "seed": 42,
    }
    main(**args)

    ckpt_path = os.path.join(run_dir, "weights", "checkpoint_last.pth")
    assert os.path.exists(ckpt_path)

    # 2. Resumed run with resume=True when checkpoint exists
    config_resumed = load_yaml_config(path=None, resume=True, output_dir=run_dir)
    assert "checkpoint" in config_resumed
    assert config_resumed["checkpoint"] == os.path.abspath(ckpt_path)

    # Run for 2 epochs using checkpoint
    args_resume = args.copy()
    args_resume["epochs"] = 2
    args_resume["checkpoint"] = config_resumed["checkpoint"]

    main(**args_resume)

    # Run directory should NOT be incremented to resume_test_run_1
    assert os.path.exists(run_dir)
    assert not os.path.exists(os.path.join(output_dir, "resume_test_run_1"))


def test_integration_migration(tmp_path):
    from publication.experiments.migrate import migrate_results

    src_dir = tmp_path / "slurm_jobs" / "exp1" / "results"
    dst_dir = tmp_path / "custom_storage" / "exp1" / "results"
    task_dir = tmp_path / "slurm_jobs" / "exp1"

    runs_dir = src_dir / "runs" / "combo1"
    runs_dir.mkdir(parents=True, exist_ok=True)
    task_dir.mkdir(parents=True, exist_ok=True)

    config_file = runs_dir / "config.yaml"
    config_file.write_text(f"output: {src_dir}/runs\ncheckpoint: {src_dir}/runs/combo1/weights/checkpoint_last.pth\n", encoding="utf-8")

    task_file = task_dir / "train_tasks.txt"
    task_file.write_text(f"mt_train --output {src_dir}/runs --name combo1\n", encoding="utf-8")

    migrate_results(src_dir=src_dir, dst_dir=dst_dir, task_dir=task_dir, move=False)

    # Check that dst_dir exists
    dst_config = dst_dir / "runs" / "combo1" / "config.yaml"
    assert dst_config.exists()
    content = dst_config.read_text(encoding="utf-8")
    assert str(dst_dir) in content
    assert str(src_dir) not in content

    # Check that task file was updated
    task_content = task_file.read_text(encoding="utf-8")
    assert str(dst_dir) in task_content
    assert str(src_dir) not in task_content


def test_orchestrate_shared_args(tmp_path, monkeypatch):
    import sys

    import yaml

    from publication.experiments import orchestrate

    cfg_file = tmp_path / "exp.yaml"
    cfg = {
        "name": "test_exp",
        "stubs": {"train": "mt_train", "eval": "mt_predict", "metric": "mt_metric"},
        "slurm": {"account": "test"},
        "datasets": {"dataset_a": {"path": str(tmp_path / "data_a"), "data_index": False}},
        "experiment": {"model": ["resnet18"], "dataset": ["dataset_a"]},
        "eval": {"dataset_a": ["dataset_a"]},
        "args": {
            "shared": {"num_workers": 12, "dtype": "float16"},
            "train": {"epochs": 5},
            "eval": {"verbose": True},
        },
    }
    cfg_file.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["orchestrate.py", str(cfg_file)])
    orchestrate.main()

    eval_tasks_file = tmp_path / "slurm_jobs" / "test_exp" / "eval_tasks.txt"
    train_tasks_file = tmp_path / "slurm_jobs" / "test_exp" / "train_tasks.txt"

    assert eval_tasks_file.exists()
    assert train_tasks_file.exists()

    eval_content = eval_tasks_file.read_text(encoding="utf-8")
    train_content = train_tasks_file.read_text(encoding="utf-8")

    assert "--num_workers 12" in eval_content
    assert "--num_workers 12" in train_content
