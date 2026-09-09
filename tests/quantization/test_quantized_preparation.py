"""Bounded native INT8 preparation preserves the existing deterministic mapping."""

import os

import pytest
import torch

pytest.importorskip("torchao")
pytest.importorskip("triton")


def check_device(device):
    if device == "cuda":
        if os.environ.get("RUN_CUDA_TESTS") != "1":
            pytest.skip("Set RUN_CUDA_TESTS=1 for preparation CUDA checks")
        if not torch.cuda.is_available():
            pytest.fail("CUDA requested but unavailable")


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("strided", [False, True])
def test_chunked_preparation_is_exact(monkeypatch, device, dtype, strided):
    check_device(device)
    from mini_trainer.modeling import _quantized_training as backend

    torch.manual_seed(42)
    values = torch.randn(17, 23, device=device, dtype=dtype)
    if strided:
        values = values.T
    values[0].zero_()
    values[1].mul_(1e-4)
    values.requires_grad_(True)
    before = values.detach().clone()
    expected_codes, expected_scales = backend.quantize_int8_rowwise(values.detach())
    monkeypatch.setattr(backend, "_PREPARATION_CHUNK_ELEMENTS", 64)
    rng = torch.cuda.get_rng_state() if device == "cuda" else torch.get_rng_state()
    result = backend.TrainingWeight.from_float(values)
    torch.testing.assert_close(result.int_data, expected_codes, rtol=0, atol=0)
    torch.testing.assert_close(result.scale, expected_scales, rtol=0, atol=0)
    torch.testing.assert_close(values, before, rtol=0, atol=0)
    torch.testing.assert_close(torch.cuda.get_rng_state() if device == "cuda" else torch.get_rng_state(), rng, rtol=0, atol=0)
    assert result.requires_grad
    assert result.dtype == values.dtype and result.device == values.device
    assert result.int_data.stride() == expected_codes.stride()
    assert result.int_data.data_ptr() != values.data_ptr()


def test_preparation_cuda_temporary_bound():
    check_device("cuda")
    from mini_trainer.modeling._quantized_training import TrainingWeight

    values = torch.randn(10000, 1280, device="cuda")
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    result = TrainingWeight.from_float(values)
    extra = torch.cuda.max_memory_allocated() - before
    assert result.shape == values.shape
    # The previous whole-matrix round/clip/cast sequence exceeds this bound.
    assert extra < 96 * 1024 * 1024
