import os

import torch.multiprocessing as mp

from mini_trainer.train import main as train_main
from tests.test_integration_train import MockBuilder, TinyMockModel


def ddp_worker(rank, world_size, tmp_path_str):
    # Set environment variables for torch.distributed env init
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29505"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)

    # Use CPU for testing
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

    try:
        train_main(**args)
    except Exception as e:
        # Re-raise so the process fails and mp.spawn detects it
        raise e


def test_integration_ddp_cpu(tmp_path):
    # Setup paths
    input_dir = str(tmp_path / "data")
    os.makedirs(input_dir, exist_ok=True)
    # create dummy class dirs so validation passes
    os.makedirs(os.path.join(input_dir, "class_a"), exist_ok=True)
    os.makedirs(os.path.join(input_dir, "class_b"), exist_ok=True)

    output_dir = str(tmp_path / "output")
    os.makedirs(output_dir, exist_ok=True)

    world_size = 2
    # Spawn 2 worker processes using torch multiprocessing
    mp.spawn(
        ddp_worker,
        args=(world_size, str(tmp_path)),
        nprocs=world_size,
        join=True,
    )

    # Verify checkponts were successfully created
    run_dir = os.path.join(output_dir, "ddp_run")
    assert os.path.exists(run_dir)
    assert os.path.isdir(run_dir)

    weights_dir = os.path.join(run_dir, "weights")
    assert os.path.exists(weights_dir)
    assert os.path.exists(os.path.join(weights_dir, "last.pt"))
    assert os.path.exists(os.path.join(weights_dir, "best.pt"))
    assert os.path.exists(os.path.join(weights_dir, "checkpoint_last.pth"))

    # Verify we don't have duplicate loggers (e.g. no tensorboard run folder for rank 1)
    tb_dir = os.path.join(run_dir, "tensorboard")
    if os.path.exists(tb_dir):
        runs = os.listdir(tb_dir)
        # Should only have 1 run folder created by rank 0
        assert len(runs) == 1
