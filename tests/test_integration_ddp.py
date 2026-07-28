import os
import socket

import torch
import torch.multiprocessing as mp

from mini_trainer.train import main as train_main
from tests.test_integration_train import MockBuilder, TinyMockModel


def _get_free_port():
    """Dynamically acquire a free localhost port to prevent test suite collisions."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return str(s.getsockname()[1])


def ddp_worker(rank, world_size, port, tmp_path_str):
    # Enable detailed PyTorch DDP diagnostics and C++ tracebacks
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    os.environ["TORCH_SHOW_CPP_STACKTRACES"] = "1"

    # Set process topology for torch.distributed
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = port
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)

    args = {
        "input": os.path.join(tmp_path_str, "data"),
        "output": os.path.join(tmp_path_str, "output"),
        "epochs": 2,
        "device": "cpu",
        "dtype": "float32",
        "name": "ddp_run",
        "builder": MockBuilder,
        "model_builder_kwargs": {"model_type": TinyMockModel(), "pretrained": False},
        "logger_builder_kwargs": {"verbose": rank == 0},
        "ema": False,
        "seed": 42,
    }

    # mp.spawn automatically captures and re-raises exceptions from child workers
    train_main(**args)


def test_integration_ddp_cpu(tmp_path):
    input_dir = str(tmp_path / "data")
    os.makedirs(os.path.join(input_dir, "class_a"), exist_ok=True)
    os.makedirs(os.path.join(input_dir, "class_b"), exist_ok=True)

    output_dir = str(tmp_path / "output")
    os.makedirs(output_dir, exist_ok=True)

    world_size = 2
    port = _get_free_port()

    # mp.spawn(..., join=True) routes worker exceptions directly back to the main thread
    mp.spawn(
        ddp_worker,
        args=(world_size, port, str(tmp_path)),
        nprocs=world_size,
        join=True,
    )

    # Assertions
    run_dir = os.path.join(output_dir, "ddp_run")
    assert os.path.isdir(run_dir)

    weights_dir = os.path.join(run_dir, "weights")
    assert os.path.exists(os.path.join(weights_dir, "last.pt"))
    assert os.path.exists(os.path.join(weights_dir, "best.pt"))
    assert os.path.exists(os.path.join(weights_dir, "checkpoint_last.pth"))

    tb_dir = os.path.join(run_dir, "tensorboard")
    if os.path.exists(tb_dir):
        assert len(os.listdir(tb_dir)) == 1

    # Test autoloading from a single .pt weights file without passing model_type or other args
    from mini_trainer.modeling import Classifier, classification_module

    loaded_model, loaded_preprocess = Classifier.build(weights=os.path.join(weights_dir, "best.pt"))
    cls_mod = classification_module(loaded_model)
    assert isinstance(cls_mod, Classifier)
    assert cls_mod.metadata["backbone_output_name"] == "fc"
    assert cls_mod.metadata["backbone_class"] == "tests.test_integration_train:TinyMockModel"
    assert loaded_preprocess is not None
    # Test that the custom preprocessing function runs
    dummy_input = torch.randn(3, 5, 5)
    processed = loaded_preprocess(dummy_input)
    assert processed.shape == (3, 5, 5)
