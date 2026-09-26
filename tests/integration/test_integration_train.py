import torch
import torchvision.transforms as tt
from torch.utils.data import DataLoader, TensorDataset

from mini_trainer.builders import BaseBuilder
from mini_trainer.logging import configure_loggers
from mini_trainer.train import main


class MockBuilder(BaseBuilder):
    @staticmethod
    def build_dataloader(batch_size, device, dtype, **kwargs):
        # Distinct class means provide a small learnable dataset.
        n = 20
        c, h, w = 3, 5, 5
        data_0 = torch.randn(n, c, h, w)
        data_1 = torch.randn(n, c, h, w) + 1.0
        data = torch.cat([data_0, data_1])
        labels = torch.cat([torch.zeros(n, dtype=torch.long), torch.ones(n, dtype=torch.long)])

        dataset = TensorDataset(data, labels)

        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        return labels.numpy(), train_loader, val_loader

    @staticmethod
    def build_augmentation(dtype):
        return tt.Compose([])

    @staticmethod
    def build_regularizer(*args, **kwargs):
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
    from mini_trainer.modeling import Classifier, classification_module

    input_dir = tmp_path / "data"
    for label in ("class_a", "class_b"):
        (input_dir / label).mkdir(parents=True)
    output_dir = tmp_path / "output"
    main(
        input=str(input_dir),
        output=str(output_dir),
        epochs=2,
        device="cpu",
        dtype="float32",
        name="test_run",
        builder=MockBuilder,
        model_builder_kwargs={"model_type": TinyMockModel(), "pretrained": False},
        logger_builder_kwargs={"verbose": True, "logger_cls": configure_loggers()},
        ema=False,
        seed=42,
    )
    run_dir = output_dir / "test_run"
    for artifact in ("config.yaml", "class_spec.json", "weights/last.pt", "weights/checkpoint_last.pth", "weights/best.pt"):
        assert (run_dir / artifact).is_file(), artifact

    # Retain the import path: checkpoints reconstruct this shared test backbone.
    loaded_model, preprocess = Classifier.build(weights=str(run_dir / "weights/best.pt"))
    head = classification_module(loaded_model)
    assert isinstance(head, Classifier)
    assert head.metadata["backbone_output_name"] == "fc"
    assert head.metadata["backbone_class"] == "tests.integration.test_integration_train:TinyMockModel"
    processed = preprocess(torch.randn(3, 5, 5))
    assert processed.shape == (3, 5, 5)
    loaded_model.eval()
    with torch.inference_mode():
        predictions = loaded_model(processed.unsqueeze(0))
    assert predictions.shape == (1, 2)
    assert torch.isfinite(predictions).all()


def test_integration_resume_train(tmp_path):
    from mini_trainer.config import load_yaml_config

    input_dir = tmp_path / "data"
    for label in ("class_a", "class_b"):
        (input_dir / label).mkdir(parents=True)
    output_dir = tmp_path / "output"
    run_name = "resume_test_run"
    run_dir = output_dir / run_name
    assert "checkpoint" not in load_yaml_config(path=None, resume=True, output_dir=str(run_dir))

    args = {
        "input": str(input_dir),
        "output": str(output_dir),
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
    checkpoint = run_dir / "weights/checkpoint_last.pth"
    assert checkpoint.is_file()
    config = load_yaml_config(path=None, resume=True, output_dir=str(run_dir))
    assert config["checkpoint"] == str(checkpoint.resolve())
    main(**{**args, "epochs": 2, "checkpoint": config["checkpoint"]})
    assert run_dir.is_dir()
    assert not (output_dir / "resume_test_run_1").exists()


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

    dst_config = dst_dir / "runs" / "combo1" / "config.yaml"
    assert dst_config.exists()
    content = dst_config.read_text(encoding="utf-8")
    assert str(dst_dir) in content
    assert str(src_dir) not in content

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
    monkeypatch.setattr(orchestrate, "create_combinations_file", lambda **kwargs: ([], []))
    orchestrate.main()

    eval_tasks_file = tmp_path / "slurm_jobs" / "test_exp" / "eval_tasks.txt"
    train_tasks_file = tmp_path / "slurm_jobs" / "test_exp" / "train_tasks.txt"
    metric_tasks_file = tmp_path / "slurm_jobs" / "test_exp" / "metric_tasks.txt"

    assert eval_tasks_file.exists()
    assert train_tasks_file.exists()
    assert metric_tasks_file.exists()

    eval_content = eval_tasks_file.read_text(encoding="utf-8")
    train_content = train_tasks_file.read_text(encoding="utf-8")
    metric_content = metric_tasks_file.read_text(encoding="utf-8")

    assert "--num_workers 12" in eval_content
    assert "--num_workers 12" in train_content
    assert "--combinations" in metric_content
