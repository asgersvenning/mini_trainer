"""Failure-mode coverage for the release input audit, without downloaded weights."""

import hashlib

import pytest
import torch

from dev.releases.mambo_v3.audit import recover_preset, verify_files


def test_integrity_rejects_same_size_corruption(tmp_path):
    path = tmp_path / "model"
    path.write_bytes(b"good")
    inventory = {"artifacts": [{"path": "model", "size": 4, "sha256": hashlib.sha256(b"good").hexdigest()}]}
    verify_files(tmp_path, inventory)
    path.write_bytes(b"evil")
    with pytest.raises(ValueError, match="integrity mismatch"):
        verify_files(tmp_path, inventory)


def test_integrity_rejects_escape(tmp_path):
    with pytest.raises(ValueError, match="escapes"):
        verify_files(tmp_path, {"artifacts": [{"path": "../outside"}]})


def test_preset_uses_explicit_indices_not_mapping_insertion_order():
    state = {
        "classifier._extra_state": {"cls2idx": {"0": {"species-c": 2, "species-a": 0, "species-b": 1}}},
        "classifier.active_indices": torch.tensor([1, 0]),
    }
    assert recover_preset(state) == ["species-b", "species-a"]
    state["classifier.active_indices"] = torch.tensor([1, 1])
    with pytest.raises(ValueError, match="Duplicate"):
        recover_preset(state)
