"""Small portable inference contracts, without downloading models or importing ORT."""

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from deployment.mambo_deploy import Predictor
from deployment.mambo_deploy.bundle import Bundle
from deployment.mambo_deploy.preprocessing import RECIPE, preprocess
from deployment.mambo_deploy.results import Prediction, hierarchy

CLASSES = {"labels": [["a", "b", "c"], ["g0", "g1"], ["f0"]], "parents": [[0, 0, 1], [0, 0]]}


@pytest.fixture
def bundle(tmp_path):
    payloads = {
        "classes.json": json.dumps(CLASSES),
        "preprocessing.json": json.dumps(RECIPE),
        "presets.json": json.dumps({"europe": {"path": "europe.classes", "count": 2}}),
        "europe.classes": "a\nb\n",
    }
    manifest = {
        "schema": "mambo-release-v1",
        "model_id": "fixture",
        "score_semantics": "hierarchical-leaf-logits-logsumexp-v1",
        "files": {},
    }
    for name, value in payloads.items():
        data = value.encode()
        (tmp_path / name).write_bytes(data)
        manifest["files"][name] = {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    (tmp_path / "release.json").write_text(json.dumps(manifest))
    return tmp_path


@pytest.fixture
def ort(monkeypatch):
    runtime = SimpleNamespace(
        SessionOptions=SimpleNamespace,
        GraphOptimizationLevel=SimpleNamespace(ORT_ENABLE_ALL=99, ORT_DISABLE_ALL=0),
        OrtValue=type("FakeOrtValue", (), {}),
        get_available_providers=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
        InferenceSession=None,
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", runtime)
    return runtime


def test_mask_recomputes_parent_scores_and_normalization():
    leaf = np.log(np.array([[0.1, 0.2, 0.7]], dtype=np.float32))
    raw, labels, indices = hierarchy(leaf, [0, 2], CLASSES)
    prediction = Prediction(raw, labels, indices)
    assert prediction[0].label == ("c", "g1", "f0")
    np.testing.assert_allclose(prediction[0].confidence, [0.875, 0.875, 1], rtol=1e-6)
    assert prediction[0].index == (1, 1, 0)
    assert prediction.global_indices.tolist() == [[[2, 1, 0]]]


def test_tiny_legacy_top1_fixture():
    import tomllib

    fixture = tomllib.loads((Path(__file__).parents[2] / "dev/releases/mambo_v3/compatibility.toml").read_text())["top1"]
    raw = [np.array([row], dtype=np.float32) for row in fixture["logits"]]
    prediction = Prediction(raw, fixture["classes"], [np.arange(len(names)) for names in fixture["classes"]])
    assert prediction[0].label == tuple(fixture["labels"])
    assert prediction[0].index == tuple(fixture["indices"])
    np.testing.assert_allclose(prediction[0].confidence, fixture["confidence"], rtol=1e-6)
    assert prediction.indices.shape == tuple(fixture["array_shape"])


def test_custom_list_replaces_preset_and_predictors_are_isolated(bundle):
    first = Predictor(bundle, class_list=["c", "c", "a"])
    second = Predictor(bundle, model="europe")
    assert first.class_list == ["a", "c"]
    assert second.class_list == ["a", "b"]
    first._apply_class_mask(-1)
    assert first.class_list == ["a", "b", "c"]
    assert second.class_list == ["a", "b"]
    with pytest.raises(ValueError, match="Unknown species"):
        Predictor(bundle, class_list=["unknown"])
    with pytest.raises(ValueError, match="empty"):
        Predictor(bundle, class_list=[])
    with pytest.raises(ValueError, match="mutually exclusive"):
        Predictor(bundle, class_list=["a"], class_mask=[0])


def test_hash_failure_and_escape_are_rejected(bundle):
    loaded = Bundle(bundle)
    with pytest.raises(ValueError, match="escapes"):
        loaded.file("../outside")
    (bundle / "europe.classes").write_text("c\nb\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        Predictor(bundle, model="europe")


def test_uint8_and_float_inputs_align_and_reject_hwc():
    image = np.arange(3 * 21 * 30, dtype=np.uint8).reshape(3, 21, 30)
    np.testing.assert_array_equal(preprocess(image), preprocess(image.astype(np.float32) / 255))
    assert preprocess(image[:1]).shape == (3, 384, 384)
    with pytest.raises(ValueError, match="CHW"):
        preprocess(image.transpose(1, 2, 0))
    with pytest.raises(ValueError, match="finite"):
        preprocess(np.full_like(image, np.nan, dtype=np.float32))


def test_batches_embeddings_and_masked_output_contract(bundle, monkeypatch):
    predictor = Predictor(bundle, batch_size=2, class_list=["c"])
    seen = []

    def fake_backend(images, embeddings):
        seen.append(len(images))
        return np.tile([1.0, 2.0, 3.0], (len(images), 1)), np.ones((len(images), 1280), dtype=np.float32) if embeddings else None

    monkeypatch.setattr(predictor, "_onnx", fake_backend)
    images = [np.zeros((3, 4, 5), dtype=np.uint8)] * 5
    result, vectors = predictor.predict_with_embeddings(images)
    assert seen == [2, 2, 1]
    assert len(result) == 5 and vectors.shape == (5, 1280)
    assert result[0].label == ("c", "g1", "f0")
    assert result[0].confidence == (1.0, 1.0, 1.0)
    assert result.metadata["bundle_sha256"] == hashlib.sha256((bundle / "release.json").read_bytes()).hexdigest()
    assert result.metadata["preprocessing_id"] == RECIPE["id"]
    with pytest.raises(ValueError, match="No images"):
        predictor.predict([])


def test_topk_serialization_handles_nested_items(tmp_path):
    classes = {"labels": [["a", "b"], ["g0", "g1"], ["f0", "f1"]], "parents": [[0, 1], [0, 1]]}
    raw, labels, indices = hierarchy(np.array([[1.0, 2.0]], dtype=np.float32), [0, 1], classes)
    result = Prediction(raw, labels, indices, topk=2)
    result.save(tmp_path / "result.json")
    assert json.loads((tmp_path / "result.json").read_text())["results"][0][0]["label"] == ["b", "g1", "f1"]
    with pytest.raises(ValueError, match="smallest retained rank"):
        Prediction(raw, labels, indices, topk=3)


def test_prediction_vocabulary_is_a_private_snapshot(tmp_path):
    raw, labels, indices = hierarchy(np.array([[1.0, 2.0, 3.0]], dtype=np.float32), [0, 2], CLASSES)
    first, second = (Prediction(raw, labels, indices) for _ in range(2))
    labels[0][0] = "changed-after-prediction"
    expected = {"0": {"a": 0, "c": 1}, "1": {"g0": 0, "g1": 1}, "2": {"f0": 0}}
    assert first.cls2idx == expected
    first.cls2idx["0"]["a"] = 99
    assert second.cls2idx == expected
    second.save(tmp_path / "result.json")
    assert json.loads((tmp_path / "result.json").read_text())["config"]["cls2idx"] == expected


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int64])
def test_confidence_preserves_readonly_scores(dtype):
    values = np.array([[1000, 999, -1000], [-3, -3, -3]], dtype=dtype)
    values.flags.writeable = False
    original = values.copy()
    result = Prediction([values] * 3, [["a", "b", "c"]] * 3, [np.arange(3)] * 3, topk=2)
    exp = np.exp(values - values.max(axis=1, keepdims=True))
    expected = exp[:, :2] / exp.sum(axis=1, keepdims=True)
    np.testing.assert_array_equal(result.confidence, np.stack([expected] * 3, axis=-1))
    np.testing.assert_array_equal(values, original)


def test_native_facade_preserves_container_and_shared_confidence(bundle, monkeypatch):
    import torch

    from mini_trainer import deploy
    from mini_trainer.hierarchical.model import HierarchicalPrediction

    monkeypatch.setattr(deploy, "_runtime", lambda: Predictor)
    facade = deploy.Predictor(device="cpu", bundle=bundle)
    # Nonnegative logits summing to one must still be treated as logits.
    raw = [np.array([[0.2, 0.8]], dtype=np.float32), np.array([[1.0]], dtype=np.float32), np.array([[1.0]], dtype=np.float32)]
    result = Prediction(raw, [["a", "b"], ["g0"], ["f0"]], [np.arange(2), np.arange(1), np.arange(1)])
    monkeypatch.setattr(facade._predictor, "predict", lambda *args, **kwargs: result)
    monkeypatch.setattr(
        facade._predictor, "predict_with_embeddings", lambda *args, **kwargs: (result, np.ones((1, 1280), dtype=np.float32))
    )
    native = facade("unused")
    assert isinstance(native, HierarchicalPrediction)
    assert native[0].label == result[0].label
    np.testing.assert_array_equal(native.confidence.numpy(), result.confidence)
    assert isinstance(native.indices, torch.Tensor)
    assert isinstance(facade.predict_with_embeddings("unused")[1], torch.Tensor)


@pytest.mark.parametrize(("precision", "use_tf32"), [("fp32", 0), ("auto", 1)])
def test_requested_cuda_rejects_cpu_only_session(bundle, monkeypatch, precision, use_tf32, ort):
    predictor = Predictor(bundle, device="cuda:0", precision=precision)
    monkeypatch.setattr(predictor.bundle, "profile", lambda key: bundle / "unused.onnx")
    captured = {}

    def session(path, sess_options, providers):
        captured["providers"] = providers
        return SimpleNamespace(disable_fallback=lambda: None, get_providers=lambda: ["CPUExecutionProvider"])

    ort.InferenceSession = session
    with pytest.raises(RuntimeError, match="refusing CPU-only fallback"):
        predictor._onnx(np.zeros((1, 3, 384, 384), dtype=np.float32), False)
    assert captured["providers"][0][1]["use_tf32"] == use_tf32


@pytest.mark.parametrize(
    ("shape", "digest"),
    [
        ((1, 1, 1), "8c0b08b2c1ddc4350fd94ea23ee0365375dae88f5cc5fa0af327ea2b35328931"),
        ((3, 2, 17), "cb8727ae82ff291103261fd12beb8cf81febaf8b449e74a7e33218d59558cc71"),
        ((3, 17, 2), "f5c44cb3c3481437b552831337baf880ac7b16017b567e199c5e219e488d23e7"),
        ((3, 383, 385), "9f33596b74bd0cf7c284ae0ed4bbede8b206724c2d8ee72dbff735bf4e5fe49e"),
        ((3, 440, 590), "37954851c9b8d10fcded680ec586885a8dcf8ffac5d7abc94d001b5245a0357a"),
        ((4, 24, 15), "e4dfd0c4a4174cd6d4913b1bf4f88bfd8c82a21802ec0d006e12594995bb1d64"),
    ],
)
def test_fp32_preprocessing_preserves_geometry_with_bounded_rounding(shape, digest):
    array = np.random.default_rng(19).integers(0, 256, size=shape, dtype=np.uint8)
    value = preprocess(array)
    assert value.dtype == np.float32 and value.flags.c_contiguous
    from deployment.mambo_deploy import preprocessing

    # Freeze the FP64 equation itself, independently of production scratch dtypes.
    image = np.ascontiguousarray(preprocessing._square(array), dtype=np.float32)
    lo, hi = preprocessing._LO, preprocessing._HI
    fraction = preprocessing._COORD - lo
    rows = image[:, lo] * (1 - fraction)[None, :, None] + image[:, hi] * fraction[None, :, None]
    pixels = rows[:, :, lo] * (1 - fraction)[None, None, :] + rows[:, :, hi] * fraction[None, None, :]
    legacy = np.rint(pixels).astype(np.float32)
    legacy /= 255
    legacy -= preprocessing._MEAN
    legacy /= preprocessing._STD
    assert hashlib.sha256(legacy.tobytes()).hexdigest() == digest
    error_in_pixel_levels = np.abs(value - legacy) * np.array(RECIPE["std"])[:, None, None] * 255
    assert error_in_pixel_levels.max() <= 1.001
    assert error_in_pixel_levels.mean() < 0.001
    np.testing.assert_array_equal(value, preprocess(array.astype(np.float32) / 255))


def test_precision_defaults_and_unsupported_combinations(bundle):
    assert Predictor(bundle).effective_precision == "fp32"
    assert Predictor(bundle, backend="torch", device="cuda").effective_precision == "fp16"
    assert Predictor(bundle, device="cuda").effective_precision == "tf32"
    assert Predictor(bundle, device="cuda", precision="fp32").effective_precision == "fp32"
    for kwargs in (
        {"precision": "fp16"},
        {"device": "cuda", "precision": "bf16"},
        {"backend": "torch", "device": "cuda", "precision": "tf32"},
    ):
        with pytest.raises(ValueError):
            Predictor(bundle, **kwargs)


def test_native_fp32_head_and_embeddings_override_outer_autocast(bundle):
    import torch

    class Head(torch.nn.Module):
        def preclassification(self, values):
            assert values.dtype == torch.float32
            assert not torch.is_autocast_enabled("cpu")
            return torch.nn.functional.normalize(values, dim=-1)

        def forward(self, values):
            return [self.preclassification(values)]

    class Model(torch.nn.Module):
        _backbone_output_name = "classifier"

        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Linear(4, 3)
            self.classifier = Head()

        def forward(self, x):
            return self.classifier(self.backbone(x))

    predictor = Predictor(bundle, backend="torch", precision="fp32")
    predictor._torch_model = Model().eval()
    original = predictor._torch_model.classifier
    values = np.ones((2, 4), dtype=np.float32)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        scores, embeddings = predictor._torch(values, True)
    assert scores.dtype == embeddings.dtype == np.float32
    assert predictor._torch_model.classifier is original
    np.testing.assert_array_equal(scores, embeddings)


@pytest.mark.parametrize("backend", ["torch", "onnx"])
def test_tta_averages_leaf_logits_before_masking_and_normalizes_embeddings(bundle, monkeypatch, backend):
    predictor = Predictor(bundle, backend=backend, tta="hflip", class_list=["a", "c"], batch_size=2, preprocess_workers=1)
    seen = []

    def runtime(images, embeddings, *, tensors=False):
        seen.append(images.copy())
        right = images[:, 0, 0, -1] > images[:, 0, 0, 0]
        scores = np.array([[4, 9, 0] if x else [0, 1, 6] for x in right], dtype=np.float32)
        vectors = np.array([[1, 0] if x else [0, 1] for x in right], dtype=np.float32)
        if tensors:
            import torch

            scores = [torch.from_numpy(scores)]
        return scores, vectors if embeddings else None

    monkeypatch.setattr(predictor, "_" + backend, runtime)
    image = np.tile(np.arange(8, dtype=np.uint8), (3, 8, 1))
    result, embedding = predictor.predict_with_embeddings([image] * 3)
    assert [len(x) for x in seen] == [2, 2, 1, 1]
    np.testing.assert_array_equal(seen[1][0], preprocess(image[..., ::-1]))
    np.testing.assert_array_equal(result.raw_logits[0], [[2, 3]] * 3)
    np.testing.assert_allclose(embedding, np.full((3, 2), 1 / np.sqrt(2)), rtol=1e-6)
    assert result.labels == predictor.predict([image] * 3).labels
    assert result.metadata["tta"] == "hflip"
    assert result[0].label == ("c", "g1", "f0")


def test_tta_rejects_undefined_embedding_and_invalid_options(bundle, monkeypatch):
    with pytest.raises(ValueError, match="tta"):
        Predictor(bundle, tta="random")
    with pytest.raises(ValueError, match="preprocess_workers"):
        Predictor(bundle, preprocess_workers=0)
    predictor = Predictor(bundle, tta="hflip", threads=2, preprocess_workers=1)
    assert predictor.threads == 2 and predictor.preprocess_workers == 1
    monkeypatch.setattr(predictor, "_onnx", lambda x, e: (np.ones((len(x), 3)), np.zeros((len(x), 2))))
    with pytest.raises(RuntimeError, match="undefined mean embedding"):
        predictor.predict_with_embeddings(np.zeros((3, 4, 4), dtype=np.uint8))


def test_tta_views_apply_before_the_unchanged_recipe():
    from deployment.mambo_deploy import TTA, View
    from deployment.mambo_deploy.augmentation import resolve_tta

    image = np.random.default_rng(31).integers(0, 256, (3, 157, 239), dtype=np.uint8)
    views = resolve_tta("five_crop").transforms
    np.testing.assert_array_equal(preprocess(views[0](image)), preprocess(image))
    assert len({preprocess(view(image)).tobytes() for view in views}) == 5
    crop = View(crop=(0, 0, 0.5, 1), quarter_turns=1)
    assert crop(image).shape == (3, 239, 78)
    assert TTA([crop]).transforms == (crop,)
    with pytest.raises(ValueError, match="crop"):
        View(crop=(0, 0, 0, 1))
    with pytest.raises(ValueError, match="callable"):
        TTA([])


@pytest.mark.parametrize(("tta", "views"), [("five_crop", 5), ("ten_crop", 10), ("d4", 8), (True, 3), ("wide_rotation_mixed_padding_5", 5)])
def test_multicrop_tta_bounds_calls_and_shares_prediction_embedding_path(bundle, monkeypatch, tta, views):
    p = Predictor(bundle, tta=tta, batch_size=2, preprocess_workers=2)
    observed = []

    def runtime(x, embeddings):
        observed.append(x.copy())
        signal = x.mean(axis=(1, 2, 3))
        return np.stack([signal, signal * 2, -signal], axis=1), np.tile(np.array([1, 0], np.float32), (len(x), 1))

    monkeypatch.setattr(p, "_onnx", runtime)
    images = [np.full((3, 20, 30), 30 + i * 50, np.uint8) for i in range(3)]
    result, vectors = p.predict_with_embeddings(images)
    assert [len(x) for x in observed] == [2] * views + [1] * views
    expected = np.mean([x.mean(axis=(1, 2, 3)) for x in observed[:views]], axis=0)
    np.testing.assert_allclose(result.raw_logits[0][:2, 0], expected, rtol=1e-6)
    assert result.labels == p.predict(images).labels
    np.testing.assert_array_equal(vectors, [[1, 0]] * 3)


def test_custom_tta_transforms_are_isolated_and_precede_preprocessing(bundle, monkeypatch):
    from deployment.mambo_deploy import TTA, View

    def darken(image):
        image[:] = 0
        return image

    p = Predictor(bundle, tta=TTA((darken, View()), name="dark-and-original"))
    seen = []

    def runtime(images, embeddings):
        seen.append(images.copy())
        return np.zeros((len(images), 3), dtype=np.float32), None

    monkeypatch.setattr(p, "_onnx", runtime)
    source = np.full((3, 7, 11), 255, dtype=np.uint8)
    result = p.predict(source)
    np.testing.assert_array_equal(source, np.full_like(source, 255))
    np.testing.assert_array_equal(seen[0][0], preprocess(np.zeros_like(source)))
    np.testing.assert_array_equal(seen[1][0], preprocess(source))
    assert result.metadata["tta"] == "dark-and-original"
    assert result.metadata["tta_views"] == 2


def test_noise_preserves_extent_channels_and_reproducibility():
    from deployment.mambo_deploy import SaltAndPepper

    source = np.full((3, 64, 96), 127, dtype=np.uint8)
    noise = SaltAndPepper(proportion=0.2, seed=42)
    first = noise(source)
    np.testing.assert_array_equal(first, noise(source))
    np.testing.assert_array_equal(first[0], first[1])
    assert first.shape == source.shape and first.dtype == np.uint8
    assert set(np.unique(first)) == {0, 127, 255}
    np.testing.assert_array_equal(source, np.full_like(source, 127))
    np.testing.assert_array_equal(SaltAndPepper(proportion=0)(source), source)
    assert not np.array_equal(first, SaltAndPepper(proportion=0.2, seed=43)(source))
    with pytest.raises(ValueError, match="proportion"):
        SaltAndPepper(proportion=-0.1)


@pytest.mark.parametrize(
    ("options", "recipe"), [([], None), (["--tta"], "rotation30_pad25_3"), (["--tta", "d4"], "d4"), (["--tta", "none"], None)]
)
def test_cli_tta_optional_recipe(bundle, monkeypatch, options, recipe):
    from deployment.mambo_deploy import cli

    class Parsed(Exception):
        pass

    def capture(*args, **kwargs):
        p = Predictor(*args, **kwargs)
        assert (p.tta.name if p.tta else None) == recipe
        raise Parsed

    monkeypatch.setattr(cli, "Predictor", capture)
    monkeypatch.setattr("sys.argv", ["mambo_predict", "-i", "example.jpg", "--bundle", str(bundle), *options])
    with pytest.raises(Parsed):
        cli.run()


@pytest.mark.parametrize(
    ("option", "settings"),
    [
        ("padded_scale", [(0, 0.08), (0, 0.15)]),
        (True, [(-30, 0.25), (30, 0.25)]),
        ("rotation30_pad25_3", [(-30, 0.25), (30, 0.25)]),
        ("wide_rotation_mixed_padding_5", [(-10, 0.15), (10, 0.15), (-30, 0.25), (30, 0.25)]),
    ],
)
def test_named_tta_preserves_released_recipe(option, settings):
    from deployment.mambo_deploy import EdgePad, RotatePad, View
    from deployment.mambo_deploy.augmentation import resolve_tta

    policy = resolve_tta(option)
    expected = (View(), *(EdgePad(padding) if degrees == 0 else RotatePad(degrees, padding) for degrees, padding in settings))
    assert policy.transforms == expected
    assert policy.name == ("rotation30_pad25_3" if option is True else option)


@pytest.mark.parametrize(
    "failure", [None, "cudaErrorNoKernelImageForDevice", "cudaErrorInvalidDeviceFunction", "CUDA out of memory", "baseline"]
)
def test_cuda_probe_profiles_and_reuse(bundle, monkeypatch, failure, ort):
    p = Predictor(bundle, device="cuda:0")
    monkeypatch.setattr(p.bundle, "profile", lambda key: bundle / key)
    sessions, calls = [], []

    def create(path, sess_options, providers):
        level = sess_options.graph_optimization_level
        sessions.append((path, level))

        def run(outputs, feed):
            calls.append((path, level, len(feed["images"])))
            assert feed["images"].dtype == np.float32
            if failure == "baseline":
                raise RuntimeError("cudaErrorNoKernelImageForDevice")
            if failure and level == 99:
                raise RuntimeError(failure)
            return [np.zeros((len(feed["images"]), 3)) for _ in outputs]

        return SimpleNamespace(
            run=run, disable_fallback=lambda: None, get_providers=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"]
        )

    ort.InferenceSession = create
    batch = np.zeros((2, 3, 384, 384), dtype=np.float32)
    if failure in ("CUDA out of memory", "baseline"):
        with pytest.raises(RuntimeError, match="out of memory" if failure != "baseline" else "baseline compatibility"):
            p._onnx(batch, False)
        assert not p._sessions
        assert len(sessions) == (2 if failure == "baseline" else 1)
        return
    if failure:
        with pytest.warns(RuntimeWarning, match="graph optimizations disabled"):
            p._onnx(batch, False)
    else:
        p._onnx(batch, False)
    before = len(calls)
    p._onnx(batch, False)
    assert len(calls) == before + 1  # No second probe on a reused session.
    assert [level for _, level in sessions] == ([99, 0] if failure else [99])
    assert p.onnx_session_info["onnx"]["profile"] == ("unoptimized" if failure else "optimized")
    assert calls[-1][2] == 2 and calls[0][2] == 1
    if not failure:
        p._onnx(batch, True)
        assert len(sessions) == 2
        assert "onnx-embedding" in p.onnx_session_info
        assert [n for _, _, n in calls[-2:]] == [1, 2]


@pytest.mark.parametrize("tta", ["none", "rotation30_pad25_3"])
@pytest.mark.parametrize("batch_size", [2, 129])
def test_streaming_api_matches_request(bundle, tmp_path, monkeypatch, tta, batch_size):
    from PIL import Image

    predictor = Predictor(bundle, batch_size=batch_size, tta=tta)

    def infer(images, embeddings=False):
        values = images.mean(axis=(2, 3))
        return values, values.copy() if embeddings else None

    monkeypatch.setattr(predictor, "_infer", infer)
    paths = []
    for i in range(5):
        path = tmp_path / f"stream-{i}.png"
        Image.fromarray(np.random.default_rng(i).integers(0, 256, (23, 29, 3), dtype=np.uint8)).save(path)
        paths.append(path)
    expected, vectors = predictor.predict_with_embeddings(paths)
    observed = list(predictor.predict_stream(iter(paths), embeddings=True, read_workers=8, prepare_workers=3))
    np.testing.assert_array_equal(np.concatenate([v for _, v in observed]), vectors)
    np.testing.assert_array_equal(np.concatenate([r.indices for r, _ in observed]), expected.indices)
    np.testing.assert_array_equal(np.concatenate([r.confidence for r, _ in observed]), expected.confidence)
    assert [len(result) for result, _ in observed] == [min(batch_size, len(paths) - start) for start in range(0, len(paths), batch_size)]
    with pytest.raises(ValueError, match="read_window must cover"):
        list(predictor.predict_stream(paths, read_window=1))


@pytest.mark.parametrize("topk", [1, 2])
@pytest.mark.parametrize(
    "raw",
    [
        pytest.param([[2, 2, -1], [-3, -3, -3], [0, 2, 1]], id="finite"),
        pytest.param([[1, np.nan, 2], [np.nan, np.nan, np.nan], [np.inf, 2, -np.inf]], id="nonfinite"),
    ],
)
def test_prediction_preserves_stable_ties_and_nonfinite_order(topk, raw):
    raw = np.array(raw, dtype=np.float32)
    with np.errstate(invalid="ignore"):
        result = Prediction([raw] * 3, [["a", "b", "c"]] * 3, [np.arange(3)] * 3, topk)
        expected = np.argsort(-raw, axis=1, kind="stable")[:, :topk]
    np.testing.assert_array_equal(result.indices, np.stack([expected] * 3, axis=-1))
    np.testing.assert_array_equal(np.isnan(result.confidence).any(axis=(1, 2)), ~np.isfinite(raw).all(axis=1))


@pytest.mark.parametrize("selected", [[0, 1, 2], [0, 2], [1]])
def test_batched_hierarchy_matches_original_and_torch(selected):
    import torch

    from deployment.mambo_deploy.results import HierarchyPlan

    leaves = np.random.default_rng(9).normal(size=(256, 3)).astype(np.float32) * 100
    plan = HierarchyPlan(selected, CLASSES)
    actual = plan.numpy(leaves)[0]
    expected = [leaves[:, selected]]
    for inverse, _, _ in plan.groups:
        grouped = np.full((len(leaves), int(inverse.max()) + 1), -np.inf, dtype=np.float32)
        for row in range(len(leaves)):
            np.logaddexp.at(grouped[row], inverse, expected[-1][row])
        expected.append(grouped)
    native = plan.torch(torch.from_numpy(leaves))[0]
    for a, b, c in zip(actual, expected, native, strict=True):
        np.testing.assert_allclose(a, b, atol=3e-5, rtol=1e-6)
        np.testing.assert_allclose(a, c, atol=3e-5, rtol=1e-6)


def test_global_native_ranks_are_reused(monkeypatch):
    import torch

    from deployment.mambo_deploy.results import HierarchyPlan
    from mini_trainer.hierarchical import utils

    plan = HierarchyPlan([0, 1, 2], CLASSES)
    native = [torch.ones((2, n)) for n in (3, 2, 1)]

    def fail(*args, **kwargs):
        pytest.fail("Global hierarchy should not be recomputed")

    monkeypatch.setattr(utils, "batched_scatter_logsumexp", fail)
    raw, _, _ = plan.torch(native[0], native)
    for values, tensor in zip(raw, native, strict=True):
        np.testing.assert_array_equal(values, tensor.numpy())


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_native_batched_preprocessing_matches_release_geometry(device):
    import torch

    from deployment.mambo_deploy.preprocessing import TorchPreprocess, prepare_batch

    if device != "cpu" and not torch.cuda.is_available():
        pytest.skip("Intentional CUDA test; set CUDA_VISIBLE_DEVICES")
    source = [
        np.random.default_rng(i).integers(0, 256, size=(3, height, width), dtype=np.uint8)
        for i, (height, width) in enumerate(((73, 125), (2048, 3072), (3072, 1024)))
    ]
    compact = prepare_batch(source, compact=True)
    assert compact.dtype == np.uint8
    tensor = TorchPreprocess(torch, device)(torch.from_numpy(compact).to(device))
    assert tensor.is_contiguous()
    actual = tensor.cpu().numpy()
    expected = prepare_batch(source)
    error = np.abs(actual - expected) * np.array(RECIPE["std"])[None, :, None, None] * 255
    assert error.max() <= 1.001
    assert error.mean() < 0.001
    np.testing.assert_array_equal(compact, prepare_batch(source, compact=True))


@pytest.mark.parametrize("compact", [False, True])
def test_virtual_tta_padding_matches_materialized_pixels(compact):
    from deployment.mambo_deploy.augmentation import EdgePad, RotatePad, View, _prepare_view
    from deployment.mambo_deploy.preprocessing import prepare_uint8

    prepare = prepare_uint8 if compact else preprocess
    for shape in [(3, 1, 7), (3, 51, 83), (3, 1025, 1537)]:
        image = np.random.default_rng(31).integers(0, 256, size=shape, dtype=np.uint8)
        original = image.copy()
        for transform in [View(), EdgePad(0), EdgePad(0.25), RotatePad(-30), RotatePad(30, 0.15)]:
            if isinstance(transform, RotatePad):
                from PIL import Image

                rotated = Image.fromarray(image.transpose(1, 2, 0)).rotate(
                    transform.degrees, resample=Image.Resampling.BILINEAR, expand=True, fillcolor=(124, 116, 104)
                )
                expected = prepare(EdgePad(transform.padding)(np.asarray(rotated).transpose(2, 0, 1)))
            else:
                expected = prepare(transform(image.copy()))
            target = np.empty_like(expected)
            actual = _prepare_view(image, transform, out=target, compact=compact)
            assert actual is target
            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(image, original)


def test_builtin_subclass_keeps_custom_transform_copy_isolation():
    from deployment.mambo_deploy.augmentation import View, _prepare_view

    class MutatingView(View):
        def __call__(self, image):
            image.fill(0)
            return image

    image = np.full((3, 9, 11), 255, dtype=np.uint8)
    actual = _prepare_view(image, MutatingView(), compact=True)
    assert not actual.any()
    assert (image == 255).all()


@pytest.mark.parametrize("format", ["JPEG", "PNG", "TIFF"])
def test_native_decode_matches_portable_inputs(tmp_path, format):
    import io

    import torch
    from PIL import Image

    from deployment.mambo_deploy.preprocessing import TorchDecode, _rgb

    decode = TorchDecode(torch)
    source = np.random.default_rng(12).integers(0, 256, (113, 179, 3), dtype=np.uint8)
    for image in [Image.fromarray(source), Image.fromarray(source[..., 0])]:
        encoded = io.BytesIO()
        image.save(encoded, format=format)
        data = encoded.getvalue()
        path = tmp_path / ("image." + format.lower())
        path.write_bytes(data)
        expected = _rgb(data)
        np.testing.assert_array_equal(decode(data), expected)
        np.testing.assert_array_equal(decode(path), expected)
        np.testing.assert_array_equal(decode(image), expected if format != "JPEG" else _rgb(image))
    np.testing.assert_array_equal(decode(source.transpose(2, 0, 1)), source.transpose(2, 0, 1))


def test_native_decode_preserves_high_bit_depth_png_conversion():
    import io

    import torch
    from PIL import Image

    from deployment.mambo_deploy.preprocessing import TorchDecode, _rgb

    source = np.array([[0, 128, 255, 256, 32768, 65535]], dtype=np.uint16)
    encoded = io.BytesIO()
    Image.fromarray(source).save(encoded, format="PNG")
    data = encoded.getvalue()
    np.testing.assert_array_equal(TorchDecode(torch)(data), _rgb(data))


def test_square_gather_preserves_pixels_across_decoded_and_strided_layouts():
    from deployment.mambo_deploy.preprocessing import prepare_uint8

    for height, width in [(73, 127), (511, 769)]:
        decoded = np.random.default_rng(81).integers(0, 256, (height, width, 3), dtype=np.uint8).transpose(2, 0, 1)
        for image in [decoded, np.ascontiguousarray(decoded), decoded[..., ::-1], decoded.transpose(0, 2, 1)]:
            for padding in [0, 0.25]:
                h, w = image.shape[1:]
                py, px = int(np.ceil(h * padding)), int(np.ceil(w * padding))
                grid = np.arange(384, dtype=np.float32)
                y = np.clip((grid * np.float32((h + 2 * py) / 384)).astype(np.intp) - py, 0, h - 1)
                x = np.clip((grid * np.float32((w + 2 * px) / 384)).astype(np.intp) - px, 0, w - 1)
                expected = image[:, y[:, None], x[None, :]]
                target = np.empty((3, 384, 384), dtype=np.uint8)
                assert prepare_uint8(image, out=target, padding=padding) is target
                np.testing.assert_array_equal(target, expected)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_cpu_interpolation_retains_reference_pixels_in_caller_storage(dtype):
    from deployment.mambo_deploy import preprocessing as p

    image = np.random.default_rng(915).integers(0, 256, (3, 275, 403), dtype=np.uint8)
    square = np.ascontiguousarray(p._square(image, padding=0.25), dtype=np.float32)
    rows = square[:, p._LO] * (1 - p._FRACTION)[None, :, None] + square[:, p._HI] * p._FRACTION[None, :, None]
    pixels = rows[:, :, p._LO] * (1 - p._FRACTION)[None, None, :] + rows[:, :, p._HI] * p._FRACTION[None, None, :]
    expected = np.rint(pixels).astype(dtype)
    expected /= 255
    expected -= p._MEAN
    expected /= p._STD
    # A view into caller-owned batch storage must also work.
    out = np.empty((3, 384, 768), dtype=dtype)[:, :, ::2]
    assert preprocess(image, out=out, padding=0.25) is out
    np.testing.assert_array_equal(out, expected)


def test_default_scope_is_global_including_legacy_facade(bundle, monkeypatch):
    assert Predictor(bundle).tta is None and Predictor(bundle, tta=False).tta is None

    from mini_trainer import deploy

    monkeypatch.setattr(deploy, "_runtime", lambda: Predictor)
    for predictor in (Predictor(bundle), deploy.Predictor(device="cpu", bundle=bundle)._predictor):
        assert predictor.preset == "full"
        assert predictor.class_list == ["a", "b", "c"]
    assert Predictor(bundle, model="europe").class_list == ["a", "b"]


@pytest.mark.parametrize("embeddings", [False, True])
def test_cli_streams_ordered_results_and_publishes_only_complete_output(bundle, tmp_path, monkeypatch, embeddings):
    import csv

    from deployment.mambo_deploy import cli

    paths = [tmp_path / "a" / f"{i}.jpg" for i in range(3)]
    closed = []
    fail = [False]

    def batches(items, **kwargs):
        assert items == paths
        assert kwargs == {"topk": 1, "embeddings": embeddings}
        try:
            for start, size in ((0, 2), (2, 1)):
                if fail[0] and start:
                    raise ValueError("decode failed")
                raw, labels, indices = hierarchy(np.tile([3.0, 2.0, 1.0], (size, 1)), [0, 1, 2], CLASSES)
                result = Prediction(raw, labels, indices, model_id="fixture", preset="full")
                yield (result, np.full((size, 1280), start, dtype=np.float32)) if embeddings else result
        finally:
            closed.append(True)

    predictor = SimpleNamespace(bundle=SimpleNamespace(classes=CLASSES), predict_stream=batches)
    monkeypatch.setattr(cli, "Predictor", lambda *args, **kwargs: predictor)
    argv = ["mambo_predict", "-i", *map(str, paths), "-o", str(tmp_path), "--name", "success"]
    if embeddings:
        argv.append("--embeddings")
    monkeypatch.setattr(sys, "argv", argv)
    cli.run()
    records = json.loads((tmp_path / "success/predictions.json").read_text())
    assert len(records["results"]) == 3 and records["metadata"]["preset"] == "full"
    with (tmp_path / "success/mini_metric.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert [r["filename"] for r in rows[::3]] == list(map(str, paths))
    assert [r["instance_id"] for r in rows[::3]] == ["0", "1", "2"]
    assert all(r["correct"] == "1" for r in rows)
    if embeddings:
        vectors = np.load(tmp_path / "success/embeddings.npy")
        assert vectors.shape == (3, 1280) and vectors.dtype == np.float32
        np.testing.assert_array_equal(vectors[:, 0], [0, 0, 2])
    argv[argv.index("success")] = "failed"
    fail[0] = True
    with pytest.raises(ValueError, match="decode failed"):
        cli.run()
    assert not (tmp_path / "failed").exists()
    assert not list(tmp_path.glob(".mambo-results-*"))
    assert len(closed) == 2
