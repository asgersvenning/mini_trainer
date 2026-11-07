import io

import pytest
import torch

from mini_trainer.utils import save_on_master
from mini_trainer.utils._concurrent import distributed


def test_save_on_master_replaces_atomically_and_keeps_the_old_file_on_interruption(tmp_path, monkeypatch):
    path = tmp_path / "checkpoint_last.pth"
    save_on_master({"epoch": 1}, path)
    save_on_master({"epoch": 2}, str(path))
    assert torch.load(path)["epoch"] == 2 and not (tmp_path / "checkpoint_last.pth.tmp").exists()

    # Killed after a partial write, then killed just before the rename: the previous checkpoint survives.
    def partial(obj, f, *args, **kwargs):
        open(f, "wb").write(b"truncated")
        raise KeyboardInterrupt

    with monkeypatch.context() as m:
        m.setattr(distributed.torch, "save", partial)
        with pytest.raises(KeyboardInterrupt):
            save_on_master({"epoch": 3}, path)
    with monkeypatch.context() as m:
        m.setattr(distributed.os, "replace", lambda *a: (_ for _ in ()).throw(KeyboardInterrupt()))
        with pytest.raises(KeyboardInterrupt):
            save_on_master({"epoch": 4}, path)
    assert torch.load(path)["epoch"] == 2
    assert sorted(p.name for p in tmp_path.iterdir()) == ["checkpoint_last.pth"]

    buffer = io.BytesIO()  # file objects are written directly
    save_on_master({"epoch": 5}, buffer)
    buffer.seek(0)
    assert torch.load(buffer)["epoch"] == 5
