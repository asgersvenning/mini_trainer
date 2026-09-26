"""Native extraction and Python fallback preserve the same dataset files."""

import shutil

import pytest

from examples import utils


@pytest.mark.parametrize("kind,tool", [("gztar", "tar"), ("zip", "unzip")])
@pytest.mark.parametrize("backend", ["native", "python", "failed-native"])
def test_extract_overwrites_files_and_preserves_nested_content(tmp_path, monkeypatch, kind, tool, backend):
    source = tmp_path / "source"
    (source / "nested/empty").mkdir(parents=True)
    (source / "nested/image.bin").write_bytes(bytes(range(256)))
    (source / "labels.txt").write_text("species\n")
    archive = shutil.make_archive(str(tmp_path / "dataset"), kind, source)
    output = tmp_path / "output"
    output.mkdir()
    (output / "labels.txt").write_text("stale")
    (output / "unrelated.txt").write_text("keep")
    if backend == "native":
        if not shutil.which(tool):
            pytest.skip(f"{tool} unavailable")
    elif backend == "python":
        monkeypatch.setattr(utils.shutil, "which", lambda _: None)
    else:
        monkeypatch.setattr(utils.shutil, "which", lambda name: name)

        def unavailable(*args, **kwargs):
            raise OSError("native extractor failed")

        monkeypatch.setattr(utils.subprocess, "Popen", unavailable)
    extract = utils.extract_tar if kind == "gztar" else utils.extract_zip
    extract(archive, output)
    assert (output / "nested/image.bin").read_bytes() == bytes(range(256))
    assert (output / "nested/empty").is_dir()
    assert (output / "labels.txt").read_text() == "species\n"
    assert (output / "unrelated.txt").read_text() == "keep"
