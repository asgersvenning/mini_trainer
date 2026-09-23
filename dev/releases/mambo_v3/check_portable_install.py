"""Qualify an installed ONNX-only runtime outside the checkout with a read-only bundle."""

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import socket
import sys
import tempfile
from pathlib import Path


def check(bundle, image):
    if importlib.util.find_spec("torch") or importlib.util.find_spec("mini_trainer"):
        raise AssertionError("Run this check in an ONNX-only environment without torch or mini_trainer")
    from mambo_deploy import Predictor
    from mambo_deploy.cli import run

    def deny_network(*args, **kwargs):
        raise AssertionError("Inference attempted a Python network connection")

    socket.create_connection = deny_network
    socket.socket.connect = deny_network
    socket.socket.connect_ex = deny_network
    original_cwd = Path.cwd()
    original_argv = sys.argv
    with tempfile.TemporaryDirectory(prefix="mambo-offline-") as directory:
        root = Path(directory)
        relocated = root / "read-only-bundle"
        shutil.copytree(bundle, relocated)
        files = [path for path in relocated.rglob("*") if path.is_file()]
        before = {str(path.relative_to(relocated)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
        try:
            for path in files:
                path.chmod(0o444)
            for path in [relocated, *[p for p in relocated.rglob("*") if p.is_dir()]]:
                path.chmod(0o555)
            os.chdir(root)
            predictor = Predictor(relocated, model="europe")
            plain = predictor.predict(image)
            embedded, vectors = predictor.predict_with_embeddings(image)
            assert plain.labels == embedded.labels and vectors.shape == (1, 1280)
            sys.argv = ["mambo_predict", "-i", str(image), "--bundle", str(relocated), "--embeddings"]
            run()
            assert (root / "results/mini_metric.csv").is_file()
            assert (root / "results/embeddings.npy").is_file()
            assert before == {str(path.relative_to(relocated)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
            return {"torch_absent": True, "relocated_read_only_bundle": True, "python_network_blocked": True, "cli": True}
        finally:
            os.chdir(original_cwd)
            sys.argv = original_argv
            for path in [relocated, *[p for p in relocated.rglob("*") if p.is_dir()]]:
                path.chmod(0o755)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("image", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = check(args.bundle.resolve(), args.image.resolve())
    args.output.write_text(json.dumps(report, indent=2) + "\n")
