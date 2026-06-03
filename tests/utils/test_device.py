import pytest
import torch

from mini_trainer.utils import (
    detect_available_devices,
    resolve_device,
    select_best_device,
    setup_device,
    validate_device,
)


def test_detect_available_devices():
    devices = detect_available_devices()
    assert "cpu" in devices
    assert isinstance(devices, list)


def test_select_best_device():
    best = select_best_device(["cpu"])
    assert best == "cpu"

    best_all = select_best_device()
    assert best_all in ["cpu", "cuda", "mps"]


def test_validate_device():
    # CPU is always valid
    validate_device("cpu")
    validate_device(torch.device("cpu"))

    # Invalid device strings
    with pytest.raises(Exception):
        validate_device("invalid_device_name")

    # CUDA device out of range
    with pytest.raises(RuntimeError):
        validate_device(f"cuda:{torch.cuda.device_count() + 10}")


def test_resolve_device():
    dev = resolve_device("cpu")
    assert isinstance(dev, torch.device)
    assert dev.type == "cpu"

    if torch.cuda.is_available():
        dev_cuda = resolve_device("cuda")
        assert dev_cuda.type == "cuda"
        assert dev_cuda.index == torch.cuda.current_device()


def test_setup_device():
    dev = setup_device("cpu")
    assert isinstance(dev, torch.device)
    assert dev.type == "cpu"
