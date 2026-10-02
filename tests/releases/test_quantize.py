"""Release pilot preserves data boundaries and records independent backend failures."""

import json
from types import SimpleNamespace

import pytest

from dev.releases.mambo_v3 import quantize


def records():
    return [
        {"set": split, "filename": f"{split}-{i}.jpg", "speciesKey": 1, "genusKey": 2, "familyKey": 3}
        for split in range(4)
        for i in range(12)
    ]


def test_sampling_preserves_splits_and_is_order_independent():
    rows = records()
    classes = {"labels": [["1"], ["2"], ["3"]]}
    small = quantize.select(rows, classes, 4, 3, 42)
    assert small == quantize.select(list(reversed(rows)) + rows, classes, 4, 3, 42)
    large = quantize.select(rows, classes, 6, 5, 42)
    for split in small:
        assert small[split] == large[split][: len(small[split])]
    assert all(r["path"].split("/")[-1][0] in "23" for r in small["train"])
    assert all(r["path"].split("/")[-1][0] == "0" for r in small["test"])
    with pytest.raises(ValueError, match="Not enough"):
        quantize.select(rows, {"labels": [["99"]]}, 1, 1, 42)
    rows[0]["filename"] = "../outside.jpg"
    with pytest.raises(ValueError, match="basename"):
        quantize.select(rows, classes, 1, 1, 42)


def test_prepare_rejects_cross_split_image_leakage(tmp_path, monkeypatch):
    from mini_trainer.integrations import parquet

    rows = [records()[0], records()[24]]
    for row in rows:
        path = tmp_path / "images" / "1" / row["filename"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"same image")
    monkeypatch.setattr(parquet, "iter_parquet", lambda _: rows)
    monkeypatch.setattr(quantize, "Bundle", lambda *a, **kw: SimpleNamespace(classes={"labels": [["1"]]}))
    args = SimpleNamespace(
        output=tmp_path / "result",
        bundle=tmp_path,
        root=tmp_path,
        metadata=tmp_path / "data.parquet",
        calibration_count=1,
        evaluation_count=1,
        threads=1,
        seed=42,
    )
    with pytest.raises(ValueError, match="Byte-identical"):
        quantize.prepare(args)
    assert not (args.output / "prepared.json").exists()


def test_all_continues_after_native_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(quantize, "prepare", lambda args: None)
    calls = []

    def run(command, **kwargs):
        calls.append(command[-1])
        return SimpleNamespace(returncode=1 if command[-1] == "torch" else 0)

    monkeypatch.setattr(quantize.subprocess, "run", run)
    monkeypatch.setattr(quantize.sys, "argv", ["quantize", "--metadata", "unused", "--root", "unused", "--output", str(tmp_path)])
    with pytest.raises(SystemExit, match="At least one backend failed"):
        quantize.main()
    report = json.loads((tmp_path / "summary.json").read_text())
    assert calls == ["torch", "onnx"]
    assert report["backends"]["torch"]["exit_code"] == 1
    assert report["backends"]["onnx"]["exit_code"] == 0
    assert report["release_ready"] is False


@pytest.mark.parametrize(
    "stage,method,folder", [("torch", "percentile", "torch"), ("onnx", "percentile", "onnx"), ("onnx", "minmax", "onnx-minmax")]
)
def test_corrupt_prepared_manifest_records_failure(tmp_path, stage, method, folder):
    (tmp_path / "prepared.json").write_text(json.dumps({"samples_sha256": "incorrect"}))
    (tmp_path / "samples.json").write_text("{}")
    args = SimpleNamespace(output=tmp_path, stage=stage, calibration_method=method, threads=1, seed=42)
    with pytest.raises(ValueError, match="manifest changed"):
        quantize.backend(args)
    report = json.loads((tmp_path / folder / "report.json").read_text())
    assert report["status"] == "failed"
    assert report["release_ready"] is False


def test_evaluation_writes_valid_paired_tables(tmp_path, monkeypatch):
    import numpy as np

    from dev.benchmarks.inference import quality_compare

    image = tmp_path / "image"
    image.write_bytes(b"image identity")
    bundle = SimpleNamespace(
        classes={"labels": [["1", "2"], ["3"], ["4"]], "parents": [[0, 0], [0]]},
        regions={"restricted": {}},
        manifest={"embedding": {"dimension": 2}},
    )
    monkeypatch.setattr(quantize, "Predictor", lambda b, model: SimpleNamespace(selected=[0, 1] if model == "full" else [1]))
    monkeypatch.setattr(quantize, "preprocess", lambda path: np.zeros((3, 2, 2), dtype=np.float32))
    checked = []

    def compare(path, output):
        manifest, _ = quality_compare.read_manifest(path)
        for mode in ("baseline", "candidate"):
            quality_compare.read_predictions(path.parent / manifest[mode]["path"], manifest[mode], manifest)
        checked.append(path.parent.name)
        return {"levels": []}

    monkeypatch.setattr(quantize, "compare", compare)
    samples = [{"path": "image", "labels": ["1", "3", "4"], "sha256": quantize.file_hash(image)}]

    def forward(images):
        return np.array([[2.0, 1.0]], dtype=np.float32), np.array([[1.0, 0.0]], dtype=np.float32)

    report = quantize.evaluate(SimpleNamespace(root=tmp_path, stage="onnx"), bundle, samples, forward, forward, tmp_path, {"seed": 42})
    assert checked == ["full", "restricted"]
    assert report["embedding_cosine"]["minimum"] == 1.0
    image.write_bytes(b"changed")
    with pytest.raises(ValueError, match="image changed"):
        quantize.evaluate(SimpleNamespace(root=tmp_path, stage="onnx"), bundle, samples, forward, forward, tmp_path, {"seed": 42})


def test_diagnostic_distinguishes_qdq_from_integer_execution(tmp_path, monkeypatch):
    import numpy as np

    image = tmp_path / "image"
    image.write_bytes(b"image")
    samples = {"test": [{"path": "image", "labels": ["a"], "sha256": quantize.file_hash(image)}]}
    quantize.write_json(tmp_path / "samples.json", samples)
    quantize.write_json(tmp_path / "release.json", {})
    quantize.write_json(
        tmp_path / "prepared.json",
        {
            "root": str(tmp_path),
            "bundle": str(tmp_path),
            "automatic_bundle": False,
            "samples_sha256": quantize.file_hash(tmp_path / "samples.json"),
            "provenance": {"bundle_sha256": quantize.file_hash(tmp_path / "release.json")},
        },
    )
    (tmp_path / "onnx").mkdir()
    quantize.write_json(tmp_path / "onnx/report.json", {"artifact_files": []})
    bundle = SimpleNamespace(
        root=tmp_path, classes={"labels": [["a", "b"]]}, manifest={"embedding": {"dimension": 2}}, profile=lambda _: "fp32"
    )
    monkeypatch.setattr(quantize, "Bundle", lambda *a, **kw: bundle)
    monkeypatch.setattr(quantize, "preprocess", lambda _: np.zeros((3, 2, 2)))

    def session(path, threads, *, profile, optimize, profile_prefix):
        integer = str(path) != "fp32" and optimize
        quantize.write_json(profile_prefix, [{"cat": "Node", "args": {"op_name": "QGemm" if integer else "Gemm"}}])
        return SimpleNamespace(
            run=lambda *a: [np.array([[0.0, 1.0]]) if integer else np.array([[1.0, 0.0]]), np.array([[1.0, 0.0]])],
            end_profiling=lambda: str(profile_prefix),
        )

    monkeypatch.setattr(quantize, "onnx_session", session)
    quantize.diagnose(SimpleNamespace(output=tmp_path, threads=4, calibration_method="percentile", stage="diagnose"))
    report = json.loads((tmp_path / "onnx-diagnostic.json").read_text())
    assert report["results"]["fp32"]["correct_species"] == 1
    assert report["results"]["qdq"]["predictions_agree_with_fp32"] == 1
    assert report["results"]["int8"]["predictions_agree_with_fp32"] == 0
    assert report["qdq_int8_prediction_agreement"] == 0
    assert report["unoptimized_qdq_has_integer_kernels"] is False


def test_minmax_cli_reuses_prepared_inputs_without_preparation(tmp_path, monkeypatch):
    captured = []
    monkeypatch.setattr(quantize, "prepare", lambda args: pytest.fail("Must reuse prepared inputs"))
    monkeypatch.setattr(quantize, "backend", lambda args: captured.append(args))
    monkeypatch.setattr(quantize.sys, "argv", ["quantize", "--stage", "onnx", "--calibration-method", "minmax", "--output", str(tmp_path)])
    quantize.main()
    assert captured[0].calibration_method == "minmax"
    assert quantize.onnx_directory(captured[0]) == tmp_path / "onnx-minmax"


def test_split_experiments_recombine_existing_graphs(tmp_path):
    import numpy as np

    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    h = onnx.helper
    calibration = tmp_path / "calibration"
    calibration.mkdir()
    files = []
    for mode in ("fp32", "qdq"):
        nodes = [h.make_node("Identity", ["images"], ["view"])]
        tensors = [onnx.numpy_helper.from_array(np.eye(2, dtype=np.float32), "model.classifier.hidden.weight")]
        incoming = "view"
        if mode == "qdq":
            tensors.extend(
                [
                    onnx.numpy_helper.from_array(np.array(1.0, dtype=np.float32), "scale"),
                    onnx.numpy_helper.from_array(np.array(0, dtype=np.uint8), "zero"),
                ]
            )
            nodes.extend(
                [
                    h.make_node("QuantizeLinear", ["view", "scale", "zero"], ["q"]),
                    h.make_node("DequantizeLinear", ["q", "scale", "zero"], ["dq"]),
                ]
            )
            incoming = "dq"
        nodes.extend(
            [
                h.make_node("Gemm", [incoming, "model.classifier.hidden.weight"], ["embedding"]),
                h.make_node("Shape", ["images"], ["original_shape"]),
                h.make_node("Reshape", ["embedding", "original_shape"], ["output_0"]),
            ]
        )

        def info(name):
            return h.make_tensor_value_info(name, onnx.TensorProto.FLOAT, [1, 2])

        graph = h.make_graph(nodes, mode, [info("images")], [info("output_0"), info("embedding")], tensors, value_info=[info("view")])
        path = calibration / ("preprocessed.onnx" if mode == "fp32" else "model.onnx")
        onnx.save(h.make_model(graph, opset_imports=[h.make_opsetid("", 18)], ir_version=10), path)
        if mode == "fp32":
            files.append({"path": str(path), "sha256": quantize.file_hash(path)})
    quantize.write_json(calibration / "report.json", {"preprocessed_files": files})
    images = [np.array([[0.1, 0.4]], dtype=np.float32)]
    samples = [{"path": "held-out", "labels": ["b"]}]
    quantize.split_evaluate(
        SimpleNamespace(threads=1, calibration_method="minmax"),
        tmp_path,
        SimpleNamespace(classes={"labels": [["a", "b"]]}),
        samples,
        images,
    )
    report = json.loads((tmp_path / "split/report.json").read_text())
    assert report["boundary"] == "view"
    assert report["results"]["fp32_backbone_fp32_head"]["correct_species"] == 1
    assert report["results"]["qdq_backbone_fp32_head"]["correct_species"] == 1
    assert report["results"]["fp32_backbone_qdq_head"]["correct_species"] == 0
    assert report["results"]["qdq_backbone_qdq_head"]["correct_species"] == 0
