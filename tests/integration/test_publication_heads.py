"""Head-weight snapshots cover every planned run and trust only checkpoints matching their records."""

import hashlib
import json

import pandas as pd
import pytest

from publication.experiments import heads


def cohort(tmp_path, name="lepi-512", weights_inside=True, recorded=None, complete=True):
    root = tmp_path / name / "study"
    attempt = root / "runs" / "full_seed42" / "attempt-000"
    (attempt / "model/weights").mkdir(parents=True)
    checkpoint = attempt / "model/weights/last.pt" if weights_inside else tmp_path / "transfer.pt"
    checkpoint.write_bytes(b"weights")
    record = {"weights_sha256": recorded or hashlib.sha256(b"weights").hexdigest()}
    (attempt / "train.json").write_text(json.dumps(record if weights_inside else record | {"weights": str(checkpoint)}))
    if complete:
        (attempt / "complete.json").write_text("{}")
    (root / "config.json").write_text(json.dumps({"parquet": "/work/global_lepi/metadata.parquet"}))
    (root / "plan.json").write_text(json.dumps([{"id": "full_seed42"}]))
    return tmp_path / name


@pytest.fixture(autouse=True)
def stub_head(monkeypatch):
    table = pd.DataFrame({"rank": [0, 0], "class_index": [0, 1], "key": ["a", "b"], "bias": [0.0, 0.0], "w0": [1.0, 0.0]})
    monkeypatch.setattr(heads, "head_table", lambda checkpoint: table.copy())


@pytest.mark.parametrize("weights_inside", [True, False])
def test_export_writes_catalogued_verified_tables_for_both_record_layouts(tmp_path, weights_inside):
    heads.export(tmp_path / "out", [cohort(tmp_path, weights_inside=weights_inside)], "test")
    catalog = pd.read_csv(tmp_path / "out/catalog.csv")
    assert catalog[["path", "kind", "study", "dataset", "run_id", "rows"]].values.tolist() == [
        ["heads/lepi-512/full_seed42.parquet", "heads", "lepi-512", "global_lepi", "full_seed42", 2]
    ]
    table = pd.read_parquet(tmp_path / "out/heads/lepi-512/full_seed42.parquet")
    assert list(table.columns[:3]) == ["study", "run_id", "rank"]
    assert set(json.loads((tmp_path / "out/manifest.json").read_text())["files"]) >= {"catalog.csv", "heads/lepi-512/full_seed42.parquet"}


@pytest.mark.parametrize(("arguments", "message"), [({"recorded": "0" * 64}, "differs"), ({"complete": False}, "not complete")])
def test_export_rejects_unverified_or_incomplete_runs(tmp_path, arguments, message):
    with pytest.raises(ValueError, match=message):
        heads.export(tmp_path / "out", [cohort(tmp_path, **arguments)], "test")
