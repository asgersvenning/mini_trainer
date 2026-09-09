import pytest

from dev.benchmarks.large_head_training import run


@pytest.mark.parametrize("hierarchical", [False, True])
def test_large_head_cpu_diagnostic_checks_full_and_frozen_updates(tmp_path, hierarchical):
    common = dict(classes=101, hierarchical=hierarchical, batch_size=2, image_size=32, warmup=1, steps=1, device="cpu", dtype="float32")
    full = run(tmp_path / "full", **common)
    frozen = run(tmp_path / "frozen", frozen=True, **common)
    assert full["status"] == frozen["status"] == "measured"
    assert full["input_sha256"] == frozen["input_sha256"]
    assert full["parameters"]["total"] == frozen["parameters"]["total"]
    assert frozen["parameters"]["trainable"] < full["parameters"]["trainable"]
    assert frozen["parameters"]["trainable"] == frozen["head_trainable_parameters"] == full["head_trainable_parameters"]
    for report in (full, frozen):
        assert report["steps"][0]["updated"]
        assert report["median_seconds_per_update"] > 0
        assert "measured_peak_allocated_bytes" not in report
    with pytest.raises(FileExistsError):
        run(tmp_path / "full", **common)


def test_native_quantized_training_is_not_reported_as_cpu_training(tmp_path):
    with pytest.raises(ValueError, match="requires CUDA"):
        run(tmp_path / "invalid", device="cpu", dtype="float32", quantized=True)
    assert not (tmp_path / "invalid").exists()
