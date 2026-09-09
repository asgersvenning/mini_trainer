import json
import os
import subprocess
import sys

import numpy as np
import pytest

from dev.benchmarks.dataset_inference import collect, inference_manifest, pair_bundle, predictions
from dev.benchmarks.inference_pair import run_pair
from dev.benchmarks.quality_compare import read_manifest, read_predictions


@pytest.fixture
def example(tmp_path):
    onnx = pytest.importorskip("onnx")
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Add", ["x", "offset"], ["leaf"]), onnx.helper.make_node("Neg", ["leaf"], ["parent"])],
        "two-input-two-level",
        [
            onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, ["batch", 2]),
            onnx.helper.make_tensor_value_info("offset", onnx.TensorProto.FLOAT, [2]),
        ],
        [onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, ["batch", 2]) for name in ("leaf", "parent")],
    )
    model = tmp_path / "model.onnx"
    onnx.save(onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 18)], ir_version=10), model)
    for i, values in enumerate(([[-2, 2], [2, -2]], [[3, -3]])):
        np.savez(tmp_path / f"batch-{i}.npz", x=np.array(values, dtype=np.float32), offset=np.zeros(2, dtype=np.float32))
    metadata = {
        "schema_version": 1,
        "split": "val",
        "provenance": {"dataset": "two-level oracle", "preprocessing": "identity"},
        "levels": [
            {"name": "leaf", "classes": ["001", "1"], "output": "leaf", "score_semantics": "logits"},
            {"name": "parent", "classes": ["a", "b"], "output": "parent", "score_semantics": "logits"},
        ],
        "samples": [
            {"instance_id": i, "filename": f"image-{i}", "labels": labs}
            for i, labs in [(7, ["001", "b"]), (3, ["001", "b"]), (9, ["1", "a"])]
        ],
        "batch_input": "x",
        "batches": [{"path": "batch-0.npz", "sample_ids": ["9", "7"]}, {"path": "batch-1.npz", "sample_ids": ["3"]}],
    }
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(metadata))
    return model, manifest, metadata


@pytest.mark.parametrize("kind", ["duplicate", "missing", "noncanonical", "output"])
def test_inference_manifest_rejects_incomplete_or_ambiguous_contract(example, kind):
    _, path, metadata = example
    if kind == "duplicate":
        metadata["batches"][1]["sample_ids"] = ["7"]
    elif kind == "missing":
        metadata["batches"].pop()
    elif kind == "noncanonical":
        metadata["batches"][1]["sample_ids"] = ["03"]
    else:
        metadata["levels"][1]["output"] = "leaf"
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        inference_manifest(path)


def test_score_semantics_and_shape_are_explicit():
    level = {"name": "leaf", "classes": ["a", "b"], "score_semantics": "logits"}
    pred, confidence = predictions(np.array([[10000, 9999]], dtype=np.float32), level, 1)
    assert pred.tolist() == [0]
    assert confidence[0] == pytest.approx(1 / (1 + np.exp(-1)))
    probability = {**level, "score_semantics": "probabilities"}
    assert predictions(np.array([[0.25, 0.75]]), probability, 1)[0].tolist() == [1]
    for values in (np.array([[1.0, 1.0]]), np.array([[-0.1, 1.1]]), np.array([[np.nan, 1]]), np.zeros((1, 3)), np.ones((1, 2), dtype=int)):
        with pytest.raises(ValueError):
            predictions(values, probability, 1)


def test_pair_pipeline_runs_real_children_and_preserves_evidence(example, tmp_path):
    pytest.importorskip("onnxruntime")
    pytest.importorskip("mini_metrics")
    model, manifest, _ = example
    output = tmp_path / "paired"
    report = run_pair(manifest, model, model, output, candidate_runtime={"save_scores": True})
    assert report["status"] == "evaluated"
    assert [stage["status"] for stage in report["stages"]] == ["inferred", "inferred", "evaluated"]
    assert report["stages"][0]["command"][0] == sys.executable
    assert all(level["prediction_changes"] == 0 for level in report["levels"])
    assert all(value == 0 for level in report["levels"] for value in level["candidate_minus_baseline"].values())
    assert (output / "candidate/scores-00001.npz").is_file()
    assert "Theil U delta" in (output / "summary.md").read_text()
    with pytest.raises(FileExistsError):
        run_pair(manifest, model, model, output)


def test_pair_pipeline_retains_failure_and_stops_before_candidate(example, tmp_path):
    model, manifest, _ = example
    output = tmp_path / "failed-pair"
    with pytest.raises(RuntimeError, match="baseline failed"):
        run_pair(manifest, model, model, output, baseline_runtime={"backend": "invalid"})
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "failed"
    assert len(report["stages"]) == 1
    assert report["stages"][0]["returncode"] != 0
    assert (output / "baseline.log").stat().st_size > 0
    assert not (output / "candidate").exists()


def test_pair_pipeline_rejects_unknown_options_before_creating_output(example, tmp_path):
    model, manifest, _ = example
    output = tmp_path / "invalid-pair"
    with pytest.raises(ValueError, match="Unknown candidate"):
        run_pair(manifest, model, model, output, candidate_runtime={"typo": True})
    assert not output.exists()


def test_cpu_collection_handles_multiple_inputs_levels_and_partial_batch(example, tmp_path):
    pytest.importorskip("onnxruntime")
    model, manifest, metadata = example
    baseline, candidate = tmp_path / "baseline", tmp_path / "candidate"
    report = collect(model, manifest, baseline, save_scores=True)
    assert report["status"] == "inferred" and len(report["batches"]) == 2
    assert [b["sample_ids"] for b in report["batches"]] == [["9", "7"], ["3"]]
    with np.load(baseline / "scores-00001.npz") as data:
        np.testing.assert_array_equal(data["leaf"], [[3, -3]])
        np.testing.assert_array_equal(data["parent"], [[-3, 3]])
    collect(model, manifest, candidate, baseline_bundle=baseline / "evaluation.json")
    pair, _ = read_manifest(candidate / "comparison.json")
    table, _ = read_predictions(candidate / "predictions.csv", pair["candidate"], pair)
    assert table["label"] == table["prediction"]
    assert len(table["label"]) == 6
    with pytest.raises(FileExistsError):
        collect(model, manifest, baseline)
    metadata["batches"][1]["sha256"] = "wrong"
    manifest.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="hash mismatch"):
        collect(model, manifest, tmp_path / "failure")
    failed = json.loads((tmp_path / "failure/report.json").read_text())
    assert failed["status"] == "failed" and len(failed["batches"]) == 1
    assert not (tmp_path / "failure/evaluation.json").exists()


def test_bundle_rejects_different_labels(example, tmp_path):
    _, _, metadata = example
    bundle = {**metadata, "artifact": {"path": "file.csv"}}
    baseline = tmp_path / "bundle.json"
    baseline.write_text(json.dumps(bundle))
    bundle["samples"][0]["labels"][0] = "changed"
    with pytest.raises(ValueError, match="samples"):
        pair_bundle(baseline, bundle)


def test_cpu_inference_to_real_mini_metrics(example, tmp_path):
    pytest.importorskip("onnxruntime")
    pytest.importorskip("mini_metrics")
    from dev.benchmarks.quality_compare import compare

    model, manifest, _ = example
    collect(model, manifest, tmp_path / "baseline")
    collect(model, manifest, tmp_path / "candidate", baseline_bundle=tmp_path / "baseline/evaluation.json")
    result = compare(tmp_path / "candidate/comparison.json", tmp_path / "metrics")
    assert all(v == pytest.approx(1) for levels in result["models"]["candidate"]["metrics"].values() for v in levels.values())


def test_cpu_collection_does_not_import_training_or_gpu_packages(example, tmp_path):
    pytest.importorskip("onnxruntime")
    model, manifest, _ = example
    code = """
import sys
class Reject:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'tensorrt', 'mini_metrics'}:
            raise AssertionError('Unnecessary CPU collection dependency: ' + fullname)
sys.meta_path.insert(0, Reject())
from dev.benchmarks.dataset_inference import collect
collect(sys.argv[1], sys.argv[2], sys.argv[3])
"""
    subprocess.run([sys.executable, "-c", code, str(model), str(manifest), str(tmp_path / "isolated")], check=True, capture_output=True)


def test_tensorrt_dataset_outputs_match_cpu_at_both_batch_sizes(example, tmp_path):
    if os.environ.get("RUN_CUDA_TESTS") != "1":
        pytest.skip("Set RUN_CUDA_TESTS=1 in an explicitly prepared TensorRT environment")
    pytest.importorskip("tensorrt")
    import torch

    assert torch.cuda.is_available(), "CUDA requested but unavailable"
    from dev.benchmarks.tensorrt_build import build

    model, manifest, _ = example
    profiles = {"x": {"min": [1, 2], "opt": [2, 2], "max": [2, 2]}, "offset": {"min": [2], "opt": [2], "max": [2]}}
    build(model, tmp_path / "batch-0.npz", tmp_path / "build", profiles=profiles, optimization=0)
    collect(model, manifest, tmp_path / "cpu", save_scores=True)
    result = collect(
        tmp_path / "build/model.engine",
        manifest,
        tmp_path / "gpu",
        backend="tensorrt",
        save_scores=True,
        baseline_bundle=tmp_path / "cpu/evaluation.json",
    )
    assert result["status"] == "inferred" and result["runtime"]["profile"] == 0
    for index in range(2):
        with np.load(tmp_path / f"cpu/scores-{index:05d}.npz") as cpu, np.load(tmp_path / f"gpu/scores-{index:05d}.npz") as gpu:
            for name in ("leaf", "parent"):
                np.testing.assert_array_equal(cpu[name], gpu[name])
