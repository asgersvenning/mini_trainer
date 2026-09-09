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
        assert report["optimizer_steps"] == 2
        assert report["settings"]["compile_optimizer"] is False
        assert "measured_peak_allocated_bytes" not in report
    with pytest.raises(FileExistsError):
        run(tmp_path / "full", **common)


def test_native_quantized_training_is_not_reported_as_cpu_training(tmp_path):
    with pytest.raises(ValueError, match="requires CUDA"):
        run(tmp_path / "invalid", device="cpu", dtype="float32", quantized=True)
    assert not (tmp_path / "invalid").exists()


@pytest.mark.parametrize("compiled,device", [(False, "cuda:0"), (True, "cpu")])
def test_invalid_optimizer_graph_requests_fail_before_output(tmp_path, compiled, device):
    with pytest.raises(ValueError, match="requires compile_optimizer|require CUDA"):
        run(tmp_path / "invalid", device=device, compile_optimizer=compiled, optimizer_cudagraphs=True)
    assert not (tmp_path / "invalid").exists()


@pytest.mark.parametrize("hierarchical", [False, True])
@pytest.mark.parametrize("compiled", [False, True])
def test_cuda_frozen_probe_releases_replaced_float_parameters(tmp_path, monkeypatch, hierarchical, compiled):
    import os
    import weakref

    import torch

    from dev.benchmarks import large_head_training as probe

    if os.environ.get("RUN_CUDA_TESTS") != "1":
        pytest.skip("Set RUN_CUDA_TESTS=1 to verify frozen INT8 parameter lifetime")
    assert torch.cuda.is_available()
    prepare = probe.prepare_quantized_training
    build_optimizer = probe.BaseBuilder.build_optimizer
    compile_optimizer = probe.prepare_compiled_optimizer
    replaced = []
    compiled_calls = []

    def inspect_compilation(optimizer, **kwargs):
        result = compile_optimizer(optimizer, **kwargs)
        assert all(getattr(optimizer, name)._mini_trainer_compiled for name in optimizer.optimizers)
        compiled_calls.append(True)
        return result

    def capture_replacements(model):
        before = {id(p): weakref.ref(p) for p in model.parameters()}
        recipe = prepare(model)
        after = {id(p) for p in model.parameters()}
        replaced.extend(ref for key, ref in before.items() if key not in after)
        return recipe

    def verify_released(*args, **kwargs):
        assert replaced
        assert all(ref() is None for ref in replaced), "Probe retains replaced floating parameters"
        return build_optimizer(*args, **kwargs)

    monkeypatch.setattr(probe, "prepare_quantized_training", capture_replacements)
    monkeypatch.setattr(probe.BaseBuilder, "build_optimizer", verify_released)
    monkeypatch.setattr(probe, "prepare_compiled_optimizer", inspect_compilation)
    result = run(
        tmp_path / "quantized",
        classes=101,
        hierarchical=hierarchical,
        batch_size=2,
        image_size=32,
        warmup=3,
        steps=1,
        frozen=True,
        quantized=True,
        dtype="bfloat16",
        compile_optimizer=compiled,
    )
    assert result["status"] == "measured"
    assert result["optimizer_steps"] == 4
    assert compiled_calls == ([True] if compiled else [])
