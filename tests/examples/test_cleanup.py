"""Dataset construction owns partial outputs, never successful or unregistered data."""

import pytest

from examples.utils import CleanupOnFailure


def test_success_retains_registered_output(tmp_path):
    output = tmp_path / "complete"
    output.write_text("data")
    with CleanupOnFailure() as cleanup:
        cleanup.register(output)
    assert output.read_text() == "data"


@pytest.mark.parametrize("error", [RuntimeError("failed"), KeyboardInterrupt(), SystemExit(1)])
def test_failure_removes_only_registered_partial_outputs(tmp_path, error):
    partial = tmp_path / "partial"
    (partial / "nested").mkdir(parents=True)
    (partial / "nested/image").write_text("partial")
    file = tmp_path / "partial-file"
    file.touch()
    retained = tmp_path / "retained"
    retained.write_text("keep")
    with pytest.raises(type(error)) as raised, CleanupOnFailure() as cleanup:
        for path in (partial, file, retained, tmp_path / "missing"):
            cleanup.register(path)
        cleanup.unregister(retained)
        raise error
    assert raised.value is error
    assert not partial.exists() and not file.exists()
    assert retained.read_text() == "keep"


def test_cleanup_error_does_not_mask_failure_or_skip_later_paths(tmp_path, monkeypatch, capsys):
    from examples import utils

    blocked, removable = tmp_path / "blocked", tmp_path / "removable"
    blocked.touch()
    removable.touch()
    remove = utils.os.remove

    def deny_one(path):
        if path == blocked:
            raise PermissionError("access denied")
        remove(path)

    monkeypatch.setattr(utils.os, "remove", deny_one)
    original = ValueError("construction failed")
    with pytest.raises(ValueError) as raised, CleanupOnFailure() as cleanup:
        cleanup.register(blocked)
        cleanup.register(removable)
        raise original
    assert raised.value is original
    assert blocked.exists() and not removable.exists()
    assert "access denied" in capsys.readouterr().out


@pytest.mark.parametrize("dangling", [False, True])
def test_registered_link_does_not_delete_its_target(tmp_path, dangling):
    target = tmp_path / "outside"
    target.mkdir()
    data = target / "image"
    data.write_text("keep")
    link = tmp_path / "partial-link"
    link.symlink_to(tmp_path / "missing" if dangling else target, target_is_directory=True)
    with pytest.raises(RuntimeError), CleanupOnFailure() as cleanup:
        cleanup.register(link)
        raise RuntimeError("construction failed")
    assert not link.is_symlink()
    assert data.read_text() == "keep"
