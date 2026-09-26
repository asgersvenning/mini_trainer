"""Historical release comparisons must not silently use different code or weights."""

from pathlib import Path

import pytest
import torch

from dev.releases.mambo_v3 import legacy_evaluation as legacy


def test_legacy_source_rejects_modified_file(tmp_path, monkeypatch):
    name = "mini_trainer/deploy.py"
    (tmp_path / name).parent.mkdir()
    (tmp_path / name).write_bytes(b"changed")
    monkeypatch.setattr(legacy.subprocess, "check_output", lambda command, **kwargs: name + "\n" if "ls-tree" in command else b"original")
    with pytest.raises(ValueError, match="Changed legacy source"):
        legacy.verify_source(tmp_path)


def test_shared_heads_allow_masks_but_reject_changed_parameters(monkeypatch):
    states = {
        legacy.FILENAMES["full"]: {"weight": torch.tensor([1.0])},
        legacy.FILENAMES["europe"]: {"weight": torch.tensor([1.0]), "classifier.active_indices": torch.tensor([0])},
        legacy.FILENAMES["north_europe"]: {"weight": torch.tensor([1.0]), "classifier.active_indices": torch.tensor([0])},
    }
    monkeypatch.setattr(legacy, "file_hash", lambda path: "verified")
    monkeypatch.setattr(legacy.tomllib, "loads", lambda text: {"artifacts": [{"path": name, "sha256": "verified"} for name in states]})
    monkeypatch.setattr(torch, "load", lambda path, **kwargs: states[path.name])
    loaded, _ = legacy.load_states(Path("unused"))
    assert len(loaded) == 3
    states[legacy.FILENAMES["europe"]]["weight"] = torch.tensor([2.0])
    with pytest.raises(ValueError, match="Learned legacy parameters differ"):
        legacy.load_states(Path("unused"))
