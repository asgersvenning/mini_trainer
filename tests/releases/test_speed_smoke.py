import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from dev.releases.mambo_v3 import speed_smoke


def metadata(tmp_path):
    path = tmp_path / "metadata.parquet"
    pq.write_table(
        pa.table(
            {
                "filename": [f"{i}.jpg" for i in range(8)],
                "set": ["0"] * 6 + ["1"] * 2,
                "speciesKey": ["species"] * 8,
                "genusKey": ["genus"] * 8,
                "familyKey": ["family"] * 8,
            }
        ),
        path,
    )
    return path


def test_sample_preserves_original_test_membership(tmp_path):
    path = metadata(tmp_path)
    records = speed_smoke.sample(path, 4)
    assert records == speed_smoke.sample(path, 4)
    assert len({r["path"] for r in records}) == 4
    assert all(int(Path(r["path"]).stem) < 6 and r["labels"] == ["species", "genus", "family"] for r in records)
    with pytest.raises(ValueError, match="Need 7 test images"):
        speed_smoke.sample(path, 7)


def test_four_variants_reuse_sample_and_route_interpreters(tmp_path, monkeypatch):
    path = metadata(tmp_path)
    original = speed_smoke.sample
    monkeypatch.setattr(speed_smoke, "sample", lambda p: original(p, 4))
    for r in original(path, 4):
        image = tmp_path / r["path"]
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(b"sample bytes")
    bundle = SimpleNamespace(root=tmp_path, profile=lambda name: None)
    monkeypatch.setattr(speed_smoke, "Predictor", lambda: SimpleNamespace(bundle=bundle))
    monkeypatch.setattr(speed_smoke.subprocess, "check_output", lambda *a, **k: "fixture")
    calls = []

    def benchmark(command, **kwargs):
        calls.append(command)
        output = Path(command[command.index("--output") + 1])
        output.mkdir()
        (output / "report.json").write_text(
            json.dumps({"cells": [{"streaming": {"images_per_second": 123}, "images_per_second": 100}], "peak_rss_kib_linux": 1024})
        )

    monkeypatch.setattr(speed_smoke.subprocess, "run", benchmark)
    args = argparse.Namespace(metadata=path, output=tmp_path / "results", onnx_python=Path("/separate-onnx/bin/python"), workers=6)
    speed_smoke.run(args)
    assert [c[0] for c in calls] == [speed_smoke.sys.executable, str(args.onnx_python)] * 2
    assert [c[c.index("--tta") + 1] for c in calls] == ["none", "none", speed_smoke.DEFAULT_TTA, speed_smoke.DEFAULT_TTA]
    assert len({c[c.index("--manifest") + 1] for c in calls}) == 1
    assert all(c[c.index("--batches") + 1] == "256" for c in calls)
    assert len((args.output / "summary.csv").read_text().splitlines()) == 5
    with pytest.raises(FileExistsError):
        speed_smoke.run(args)


def test_automatic_runtime_installation_stays_outside_active_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("MAMBO_CACHE", str(tmp_path))
    calls = []

    def install(command, **kwargs):
        calls.append(command)
        if command[1] == "venv":
            interpreter = Path(command[-1]) / "bin/python"
            interpreter.parent.mkdir(parents=True)
            interpreter.touch()

    monkeypatch.setattr(speed_smoke.subprocess, "run", install)
    interpreter = speed_smoke.prepare_onnx_runtime()
    assert interpreter.is_relative_to(tmp_path)
    assert str(interpreter) != speed_smoke.sys.executable
    assert calls[-1][calls[-1].index("--python") + 1] == str(interpreter)
    assert "onnxruntime-gpu[cuda,cudnn]==1.22.0" in calls[-1]
    calls.clear()
    assert speed_smoke.prepare_onnx_runtime() == interpreter
    assert all(c[1:3] == ["pip", "install"] for c in calls)
