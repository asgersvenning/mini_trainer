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
            [
                sys.executable,
                "-m",
                "mini_trainer.visualization.prototype_space",
                "--no-browser",
                "--pca-only",
                "--cache-dir",
                str(tmp_path / "cache"),
            ],
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


def test_analysis_cache_reuses_data_and_refreshes_assets(tmp_path, monkeypatch):
    from mini_trainer.visualization.prototype_space import explore
    from mini_trainer.visualization.prototype_space.launch import generate

    weights = tmp_path / "weights.pt"
    checkpoint(weights)
    cache = tmp_path / "cache"
    first = generate(weights, tmp_path / "first", angular=False, cache_dir=cache)
    reference = first.with_name("report-data.json").read_bytes()

    def forbidden(*args, **kwargs):
        pytest.fail("cache hit must not analyze")

    monkeypatch.setattr(explore, "create_report", forbidden)
    second = generate(weights, tmp_path / "second", angular=False, cache_dir=cache)
    assert second.with_name("report-data.json").read_bytes() == reference
    assert "captureViewerState" in second.read_text()
    assert "__STATE_SCRIPT__" not in second.read_text()


def test_analysis_cache_invalidates_and_rejects_partial_or_corrupt_entries(tmp_path):
    from mini_trainer.visualization.prototype_space.cache import analysis_key, read_analysis, store_analysis

    weights = tmp_path / "weights"
    weights.write_bytes(b"model one")
    options = {"angular": False, "synthetic": False, "threads": 4}
    key, contract = analysis_key(weights, **options)
    assert key != analysis_key(weights, **{**options, "angular": True})[0]
    weights.write_bytes(b"model two")
    assert key != analysis_key(weights, **options)[0]
    root, output = tmp_path / "cache", tmp_path / "output"
    output.mkdir()
    for name in ("report-data.json", "summary.json"):
        (output / name).write_text('{"valid": true}')
    assert read_analysis(root, key, contract) is None
    store_analysis(root, key, contract, output)
    assert read_analysis(root, key, contract) is not None
    (root / key / "report-data.json").write_text('{"altered": true}')
    assert read_analysis(root, key, contract) is None
    store_analysis(root, key, contract, output)
    assert read_analysis(root, key, contract) is not None
    store_analysis(root, key, contract, output, max_bytes=0)
    assert read_analysis(root, key, contract) is None


def test_cancelled_job_cannot_publish_after_replacement(tmp_path, monkeypatch):
    import threading
    from types import SimpleNamespace

    from mini_trainer.visualization.prototype_space import launch

    entered, release = threading.Event(), threading.Event()
    processes = []

    class Process:
        def __init__(self, command, **kwargs):
            self.output = launch.Path(command[command.index("--output") + 1])
            self.output.mkdir()
            self.code = None
            self.first = not processes
            processes.append(self)

        def poll(self):
            return self.code

        def terminate(self):
            self.code = -15

        def wait(self, timeout=None):
            if self.first and timeout is None:
                entered.set()
                release.wait(5)
            if self.code is None:
                self.code = 0
            (self.output / "report-data.json").write_text(json.dumps({"case": {"metadata": {}, "names": ["1"]}}))
            (self.output / "explorer.html").write_text("old" if self.first else "new")
            return self.code

    monkeypatch.setattr(launch.subprocess, "Popen", Process)
    workspace = launch.Workspace(tmp_path)
    workspace.service = SimpleNamespace(allowed_ids=set())
    workspace.start(tmp_path / "old.pt")
    assert entered.wait(5)
    workspace.start(tmp_path / "new.pt")
    workspace.workers[-1].join(5)
    assert workspace.status()["state"] == "ready"
    release.set()
    workspace.close()
    assert (workspace.public / "explorer.html").read_text() == "new"
    assert workspace.ready
    assert not (tmp_path / "job-1").exists()
    assert all(not worker.is_alive() for worker in workspace.workers)


def test_view_state_survives_new_session_and_cache_clear(tmp_path):
    from mini_trainer.visualization.prototype_space.cache import clear_analysis_cache
    from mini_trainer.visualization.prototype_space.launch import Workspace

    cache = tmp_path / "cache"
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    before = Workspace(first, cache_dir=cache)
    state = {"case": "Production checkpoint", "view": {"version": 1, "selected": "42"}}
    before.write_view_state("model-hash", state)
    clear_analysis_cache(cache)
    after = Workspace(second, cache_dir=cache)
    assert after.token != before.token
    assert after.read_view_state("model-hash") == state
    assert after.read_view_state("other-model") is None
    after.write_view_state("model-hash", None)
    assert before.read_view_state("model-hash") is None


def test_cancel_terminates_real_analysis_process(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from mini_trainer.visualization.prototype_space import launch

    original = subprocess.Popen
    processes = []

    def sleeping_analysis(command, **kwargs):
        process = original([sys.executable, "-c", "import time; time.sleep(60)"], **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(launch.subprocess, "Popen", sleeping_analysis)
    workspace = launch.Workspace(tmp_path)
    workspace.service = SimpleNamespace(allowed_ids=set())
    workspace.start(tmp_path / "weights.pt")
    try:
        deadline = time.monotonic() + 5
        while not processes and time.monotonic() < deadline:
            time.sleep(0.01)
        assert processes and processes[0].poll() is None
        workspace.cancel()
        workspace.workers[0].join(5)
        assert processes[0].poll() is not None
        assert workspace.status()["state"] == "cancelled"
        assert not (tmp_path / "job-1").exists()
        assert not (workspace.public / "explorer.html").exists()
    finally:
        workspace.close()
