"""Small real-image backend contract check; not a performance or accuracy benchmark."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np


def qualify(bundle, dataset, device, backends, tta="none"):
    from mambo_deploy import Predictor

    if "torch" in backends:
        import torch

        torch.set_num_threads(2)
    images = []
    for directory in sorted(dataset.iterdir()):
        if directory.is_dir() and (paths := sorted(directory.glob("*.jpg"))):
            images.append(paths[0])
        if len(images) == 4:
            break
    if len(images) != 4:
        raise ValueError("Need four species directories containing JPEGs")
    report = {"device": device, "tta": tta, "images": [], "variants": {}, "purpose": "bounded contract qualification; not a benchmark"}
    for path in images:
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        report["images"].append({"path": str(path), "sha256": digest})
    custom = None
    for backend in backends:
        predictor = Predictor(bundle, backend=backend, device=device, model="full", batch_size=2, tta=tta)
        plain = predictor.predict(images)
        embedded, vectors = predictor.predict_with_embeddings(images)
        if plain.labels != embedded.labels:
            raise AssertionError(f"{backend}: prediction-only and embedding graphs disagree on top-1")
        if vectors.shape != (4, 1280) or not np.isfinite(vectors).all():
            raise AssertionError("Embedding shape/finite contract failed")
        np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4)
        # Reuse loaded models while testing a regional mask and a custom list.
        europe = predictor.bundle.file(predictor.bundle.regions["europe"]["path"]).read_text().splitlines()
        predictor._apply_class_mask(europe)
        regional = predictor.predict(images)
        if any(item.label[0] not in europe for item in regional):
            raise AssertionError("Regional filtering failed")
        if custom is None:
            custom = list(dict.fromkeys(item.label[0] for item in plain))
        predictor._apply_class_mask(custom)
        limited, limited_vectors = predictor.predict_with_embeddings(images)
        if any(item.label[0] not in custom for item in limited):
            raise AssertionError("Custom filtering failed")
        np.testing.assert_array_equal(vectors, limited_vectors)
        report["variants"][backend] = {
            "full": plain.to_dict(),
            "europe": regional.to_dict(),
            "custom": limited.to_dict(),
            "embedding_shape": list(vectors.shape),
            "prediction_modes_agree": True,
        }
        if backend == "onnx":
            report["variants"][backend]["onnx_session_info"] = predictor.onnx_session_info
            report["variants"][backend]["providers"] = {name: session.get_providers() for name, session in predictor._sessions.items()}
    if len(backends) == 2:
        for mode in ("full", "europe", "custom"):
            a, b = (report["variants"][backend][mode] for backend in backends)
            report[f"{mode}_backend_top1_agreement"] = sum(x["label"] == y["label"] for x, y in zip(a, b)) / len(a)
    report["versions"] = {}
    for name in ("numpy", "pillow", "torch", "onnxruntime", "onnxruntime-gpu", "mambo-deploy"):
        try:
            report["versions"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--backends", nargs="+", choices=["torch", "onnx"], default=["torch", "onnx"])
    parser.add_argument("--tta", default="none")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = qualify(args.bundle, args.dataset, args.device, args.backends, args.tta)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)
