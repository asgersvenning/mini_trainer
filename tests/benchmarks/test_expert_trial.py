"""Offline coverage for the bounded staging helper."""

import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("expert_trial", Path(__file__).parents[2] / "dev/ucloud/expert_trial.py")
trial = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trial)


def test_stage_preserves_unknown_folder_and_bytes(tmp_path):
    source = tmp_path / "source"
    for name in ("111", "999"):
        (source / name).mkdir(parents=True)
        (source / name / "image.jpg").write_bytes(b"image bytes")
        (source / name / "notes.txt").write_text("not selected")
    output = tmp_path / "output"
    output.mkdir()
    destination = tmp_path / "staged"
    trial.stage(source, destination, output, 2, 1024, 2)
    report = json.loads((output / "staging.json").read_text())
    assert len(report["files"]) == 2
    for name in ("111", "999"):
        assert (destination / name / "image.jpg").read_bytes() == b"image bytes"
    assert len(trial.select_images(source, 1)) == 1


def test_stage_rejects_byte_budget_before_copy(tmp_path):
    source = tmp_path / "source"
    (source / "999").mkdir(parents=True)
    (source / "999/image.jpg").write_bytes(b"too large")
    with pytest.raises(RuntimeError, match="limit"):
        trial.stage(source, tmp_path / "staged", tmp_path, 1, 1, 1)
    assert not (tmp_path / "staged").exists()
