"""Publishing a browser bundle must preserve model identity and be atomic."""

import hashlib
import json

import pytest

from mini_trainer.visualization.prototype_space.browser import package_bundle
from tests.utils.test_prototype_launcher import checkpoint


def test_browser_bundle_checks_identity_and_cleans_failed_publication(tmp_path):
    weights = tmp_path / "weights.pt"
    checkpoint(weights)
    export = tmp_path / "onnx"
    export.mkdir()
    (export / "model.onnx").write_bytes(b"fixture graph")
    manifest = {
        "output_semantics": "predictions_and_preclassification_embedding",
        "classifiers": [{"metadata": {"cls2idx": {"alpha": 2, "beta": 0, "gamma": 3, "delta": 1}}}],
        "source": {"checkpoint_sha256": hashlib.sha256(weights.read_bytes()).hexdigest()},
        "input": {"dtype": "float32", "shape": ["batch", 3, 384, 384]},
        "outputs": [{"name": "output_0"}, {"name": "embedding"}],
        "preprocessing": {"recipe": {"contract": "nearest-square-uint8-bilinear-center-imagenet-v1", "size": 384, "resize": 438}},
        "artifacts": {"model.onnx": hashlib.sha256(b"fixture graph").hexdigest()},
    }
    (export / "manifest.json").write_text(json.dumps(manifest))
    runtime = tmp_path / "runtime"
    (runtime / "dist").mkdir(parents=True)
    (runtime / "package.json").write_text(json.dumps({"version": "1.24.3"}))
    for name in ["ort.wasm.min.js", "ort-wasm-simd-threaded.mjs", "ort-wasm-simd-threaded.wasm"]:
        (runtime / "dist" / name).write_bytes(b"fixture runtime")
    output = tmp_path / "published"
    with pytest.raises(FileNotFoundError):
        package_bundle(export, output, runtime, weights)
    assert not output.exists()
    assert not list(tmp_path.glob(".browser-bundle-*"))
    (runtime / "LICENSE.txt").write_text("fixture license")
    assert package_bundle(export, output, runtime, weights) == output
    bundle = json.loads((output / "manifest.json").read_text())
    assert bundle["classes"] == [["beta", "delta", "alpha", "gamma"]]
    assert (output / "runtime/ort-wasm-simd-threaded.js").exists()
    assert (output / "prototypes.f32").stat().st_size == 4 * 5 * 4
    for name, digest in bundle["artifacts"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest
    with pytest.raises(FileExistsError):
        package_bundle(export, output, runtime, weights)
    (export / "model.onnx").write_bytes(b"corrupted graph")
    with pytest.raises(ValueError, match="Invalid export artifact"):
        package_bundle(export, tmp_path / "corrupt", runtime, weights)
