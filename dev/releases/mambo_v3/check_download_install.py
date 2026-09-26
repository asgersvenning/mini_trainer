"""Check the installed default model download, global scope, embeddings and offline reuse.

Run with a fresh MAMBO_CACHE: python -I check_download_install.py IMAGE OUTPUT_JSON.
"""

import hashlib
import importlib.metadata
import json
import os
import sys
from pathlib import Path

import numpy as np
from mambo_deploy import Predictor, download

image, output = map(Path, sys.argv[1:])
cache = os.environ.get("MAMBO_CACHE")
if not cache or (Path(cache).exists() and any(Path(cache).iterdir())):
    raise RuntimeError("Set MAMBO_CACHE to a new or empty directory to qualify actual downloads")
urls = []
original = download.urlopen


def fetch(url, **kwargs):
    urls.append(url)
    return original(url, **kwargs)


download.urlopen = fetch
predictor = Predictor(batch_size=2)
assert predictor.preset == "full"
plain = predictor.predict(image)
embedded, vectors = predictor.predict_with_embeddings(image)
assert plain.labels == embedded.labels
assert vectors.shape == (1, 1280) and np.isfinite(vectors).all()
np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4)
if not urls:
    raise RuntimeError("No model assets were downloaded; automatic download was not qualified")
os.environ["MAMBO_OFFLINE"] = "1"


def deny(*args, **kwargs):
    raise AssertionError("offline request attempted a download")


download.urlopen = deny
second = Predictor()
assert second.predict(image).labels == plain.labels
report = {
    "automatic_download": True,
    "urls": urls,
    "offline_reuse": True,
    "default_scope": predictor.preset,
    "model_id": predictor.bundle.manifest["model_id"],
    "bundle_sha256": predictor.bundle.manifest_sha256,
    "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
    "embeddings_shape": list(vectors.shape),
    "packages": {n: importlib.metadata.version(n) for n in ("mambo-v3", "onnxruntime", "numpy", "pillow")},
}
output.write_text(json.dumps(report, indent=2) + "\n")
print("PASS: actual public downloads, global defaults, embeddings and offline reuse")
