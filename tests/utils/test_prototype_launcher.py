"""Reusable explorer exports and browser-selected checkpoints."""

import json
import re
import subprocess
import sys
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest
import torch
from torch import nn

pytest.importorskip("scipy")


def checkpoint(path):
    layer = nn.Linear(5, 4)
    state = {f"classifier.linear.{key}": value for key, value in layer.state_dict().items()}
    state["classifier._extra_state"] = {
        "normalized": False,
        "classifier_class": "mini_trainer.modeling.classifier:Classifier",
        "cls2idx": {"alpha": 2, "beta": 0, "gamma": 3, "delta": 1},
    }
    torch.save(state, path)
    return layer


def test_export_public_api_and_assets(tmp_path):
    from mini_trainer.visualization.prototype_space.explore import load_prototypes
    from mini_trainer.visualization.prototype_space.launch import generate

    path = tmp_path / "weights.pt"
    layer = checkpoint(path)
    weight, names, _, _ = load_prototypes(path)
    torch.testing.assert_close(weight, layer.weight)
    assert names == ["beta", "delta", "alpha", "gamma"]
    html = generate(path, tmp_path / "report", angular=False)
    assert html.is_file()
    assert "__REPORT_DATA__" not in html.read_text()
    assert "function animateThumbnails" in html.read_text()
    report = json.loads(html.with_name("report-data.json").read_text())
    assert list(report) == ["Production checkpoint"]
    assert report["Production checkpoint"]["names"] == names


def test_browser_upload_and_recovery(tmp_path):
    checkpoint_path = tmp_path / "weights.pt"
    checkpoint(checkpoint_path)
    log = tmp_path / "server.log"
    with log.open("w") as output:
        process = subprocess.Popen(
            [sys.executable, "-m", "mini_trainer.visualization.prototype_space", "--no-browser", "--pca-only"],
            stdout=output,
            stderr=subprocess.STDOUT,
        )
    try:
        deadline = time.monotonic() + 30
        address = None
        while time.monotonic() < deadline:
            match = re.search(r"http://localhost:\d+/", log.read_text())
            if match:
                address = match.group()
                break
            assert process.poll() is None, log.read_text()
            time.sleep(0.05)
        assert address, log.read_text()
        with urlopen(address) as response:
            page = response.read().decode()
        token = re.search(r"'X-Explorer-Token':'([^']+)'", page).group(1)
        with pytest.raises(HTTPError) as rejected:
            urlopen(Request(address + "api/weights", data=b"bad"))
        assert rejected.value.code == 403

        def upload(content):
            with urlopen(Request(address + "api/weights", data=content, headers={"X-Explorer-Token": token})) as response:
                assert response.status == 202
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                with urlopen(address + "api/session") as response:
                    status = json.load(response)
                if status["state"] in {"ready", "error"}:
                    return status
                time.sleep(0.1)
            pytest.fail("generation timed out")

        assert upload(b"not a checkpoint")["state"] == "error"
        status = upload(checkpoint_path.read_bytes())
        assert status["state"] == "ready", status
        with urlopen(address + "report-data.json") as response:
            data = json.load(response)
        assert data["Production checkpoint"]["names"] == ["beta", "delta", "alpha", "gamma"]
        with pytest.raises(HTTPError) as missing:
            urlopen(address + "selected-weights.pt")
        assert missing.value.code == 404
    finally:
        process.terminate()
        process.wait(timeout=10)
