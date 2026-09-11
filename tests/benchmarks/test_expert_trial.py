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


def test_reuse_requires_complete_intact_staging(tmp_path):
    staged = tmp_path / "ram"
    staged.mkdir()
    image = staged / "image.jpg"
    image.write_bytes(b"image")
    previous = tmp_path / "previous"
    previous.mkdir()
    (previous / "inference.yaml").write_text(json.dumps({"input": str(staged)}))
    with pytest.raises(FileNotFoundError):
        trial.reuse_stage(previous)
    (previous / "staging.json").write_text(json.dumps({"files": [{"staged": str(image), "bytes": 5}]}))
    assert trial.reuse_stage(previous)[0] == staged
    image.write_bytes(b"truncated")
    with pytest.raises(RuntimeError, match="manifest"):
        trial.reuse_stage(previous)


def test_index_staging_preserves_only_test_rows_and_labels(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    for name in ("a.jpg", "b.jpg"):
        (source / name).write_bytes(name.encode())
    index = source / "data_index.json"
    index.write_text(
        json.dumps(
            {
                "path": ["missing-train.jpg", "b.jpg", "missing-val.jpg", "a.jpg"],
                "split": ["train", "test", "validation", "test"],
                "label": [[1, 2], [999, 3], [4, 5], [888, 6]],
                "class": [[0, 0], [-1, 1], [2, 2], [-1, 3]],
            }
        )
    )
    output = tmp_path / "output"
    output.mkdir()
    destination = tmp_path / "staged"
    trial.stage(source, destination, output, 1, 1024, 2, index)
    staged = json.loads((output / "staged-index.json").read_text())
    assert staged["split"] == ["test", "test"]
    assert staged["label"] == [[999, 3], [888, 6]]
    assert staged["class"] == [[-1, 1], [-1, 3]]
    assert [Path(p).read_bytes() for p in staged["path"]] == [b"b.jpg", b"a.jpg"]
    assert json.loads((output / "staging.json").read_text())["scope"] == "supplied_test_split"


def test_index_rejects_inconsistent_columns(tmp_path):
    index = tmp_path / "index.json"
    index.write_text(json.dumps({"path": ["a.jpg"], "split": [], "label": [1]}))
    with pytest.raises(ValueError, match="lengths"):
        trial.test_rows(index)
