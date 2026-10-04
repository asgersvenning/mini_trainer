"""Research downloads must be complete, verified and confined to their cache."""

import io

import pytest

from publication.experiments import artifacts


def test_artifact_roundtrip_and_corrupt_download(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "predictions.csv").write_bytes(b"retained evidence\n")
    manifest = artifacts.create(source, ["predictions.csv"], "pinned-revision")
    cache = tmp_path / "cache"
    calls = []

    def download(url, **kwargs):
        calls.append(url)
        return io.BytesIO(b"retained evidence\n")

    monkeypatch.setattr(artifacts, "urlopen", download)
    artifacts.restore(manifest, cache, "https://archive.example/share")
    artifacts.restore(manifest, cache, "https://archive.example/share")
    assert len(calls) == 1
    (cache / "predictions.csv").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="corrupt"):
        artifacts.restore(manifest, cache)
    monkeypatch.setattr(artifacts, "urlopen", lambda *a, **kw: io.BytesIO(b"truncated"))
    with pytest.raises(ValueError, match="checksum"):
        artifacts.restore(manifest, cache, "https://archive.example/share")
    assert (cache / "predictions.csv").read_bytes() == b"corrupt"
    assert not list(cache.glob("*.part"))


@pytest.mark.parametrize("name", ["../outside", "/absolute", "a/../../outside", "a\\outside", "./alias"])
def test_artifact_paths_cannot_escape(tmp_path, name):
    with pytest.raises(ValueError, match="path"):
        artifacts.artifact_path(tmp_path, name)


def test_artifact_symlink_cannot_escape(tmp_path):
    root = tmp_path / "cache"
    root.mkdir()
    (root / "escape").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes"):
        artifacts.artifact_path(root, "escape/evidence")


def test_saved_predictions_are_paired_and_unthresholded(tmp_path):
    from publication.experiments.statistics.saved_predictions import summarize

    path = tmp_path / "predictions.csv"
    header = "filename,level,label,prediction,threshold,prediction_made,known_label\n"
    rows = "old/images/a.jpg,0,a,a,0,1,1\nold/images/b.jpg,0,b,a,0,1,1\nold/images/c.jpg,0,b,b,0,1,1\n"
    path.write_text(header + rows)
    expected = {"images/a.jpg": "a", "images/b.jpg": "b", "images/c.jpg": "b"}
    metrics, classes, pairs = summarize(path, "images/", expected)
    assert metrics["macro_recall"] == pytest.approx(0.75)
    assert metrics["micro_recall"] == pytest.approx(2 / 3)
    assert sum(row["count"] for row in pairs) == 3
    with pytest.raises(ValueError, match="identities"):
        summarize(path, "images/", {"images/a.jpg": "a"})
    path.write_text(header + rows + rows.splitlines()[0] + "\n")
    with pytest.raises(ValueError, match="Duplicate"):
        summarize(path, "images/")
    path.write_text(header + rows.replace(",0,1,1", ",0.5,1,1"))
    with pytest.raises(ValueError, match="unthresholded"):
        summarize(path)
