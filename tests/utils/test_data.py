import os
import json
import pytest
from mini_trainer.utils.data import find_images, get_metadata, parse_class_spec

def test_find_images(tmp_path):
    d = tmp_path / "images"
    d.mkdir()
    (d / "img1.jpg").write_bytes(b"\xff\xd8\xff")
    (d / "img2.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (d / "not_img.txt").write_text("hello")
    
    images = find_images(str(d))
    assert len(images) == 2
    basenames = sorted([os.path.basename(p) for p in images])
    assert basenames == ["img1.jpg", "img2.png"]

def test_get_metadata(tmp_path):
    p = tmp_path / "meta.json"
    data = {
        "path": ["a", "b"],
        "class": [0, 1],
        "split": ["train", "validation"]
    }
    with open(p, "w") as f:
        json.dump(data, f)
        
    meta = get_metadata(str(p))
    # It converts to numpy arrays
    assert len(meta["path"]) == 2
    assert meta["split"][0] == "train"
    
    with pytest.raises(FileNotFoundError):
        get_metadata(str(tmp_path / "nonexistent.json"))

def test_parse_class_spec(tmp_path):
    d = tmp_path / "classes"
    d.mkdir()
    (d / "cat").mkdir()
    (d / "dog").mkdir()
    (d / "file.txt").touch()
    
    spec = parse_class_spec(dir=str(d))
    assert spec["num_classes"] == 2
    assert spec["cls2idx"] == {"cat": 0, "dog": 1}
    
    # Save/Load
    p = tmp_path / "spec.json"
    parse_class_spec(path=str(p), dir=str(d))
    loaded = parse_class_spec(path=str(p))
    assert loaded == spec
