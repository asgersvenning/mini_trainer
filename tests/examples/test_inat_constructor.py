import json

import pytest

from examples.inat2021.construct import build_data_index


def category(genus, prefix="00000"):
    return f"{prefix}_Animalia_Arthropoda_Insecta_Lepidoptera_Family_{genus}_species"


def write_image(root, relative):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(relative.encode())


@pytest.mark.parametrize("collision", ["directory", "file", "planned-target"])
def test_taxonomy_collision_preserves_all_sources_before_renaming(tmp_path, collision):
    taxonomy = {"Early species": ["101", "201"], "Late species": ["102", "202"]}
    (tmp_path / "taxonomy_map.json").write_text(json.dumps(taxonomy))
    write_image(tmp_path, f"train_mini/{category('Early')}/first.jpg")
    write_image(tmp_path, f"val/{category('Late')}/second.jpg")
    if collision == "directory":
        write_image(tmp_path, "val/102/existing.jpg")
    elif collision == "file":
        (tmp_path / "val/102").write_bytes(b"existing file")
    else:
        write_image(tmp_path, f"val/{category('Late', '00001')}/third.jpg")
    (tmp_path / "data_index.json").write_text("previous index")
    before = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    with pytest.raises(FileExistsError, match="102"):
        build_data_index(str(tmp_path), "train_mini", "val")
    after = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    assert after == before


@pytest.mark.parametrize("splits", [("train_mini", "val"), ("train_mini",), ("val",)])
def test_index_retains_split_order_taxonomy_and_legacy_numeric_folders(tmp_path, splits):
    taxonomy = ["101", "201", "301", "401", "501", "601", "701"]
    (tmp_path / "taxonomy_map.json").write_text(json.dumps({"Known species": taxonomy}))
    for split in splits:
        write_image(tmp_path, f"{split}/{category('Known')}/b.PNG")
        write_image(tmp_path, f"{split}/{category('Known')}/a.jpg")
        write_image(tmp_path, f"{split}/999/unmapped.jpeg")
        (tmp_path / split / category("Known") / "ignored.txt").write_text("not an image")
    build_data_index(str(tmp_path), "train_mini", "val")
    expected = {"path": [], "split": [], "label": []}
    for split in splits:
        expected["path"].extend(f"{split}/{name}" for name in ("101/a.jpg", "101/b.PNG", "999/unmapped.jpeg"))
        expected["split"].extend(["train" if split == "train_mini" else "validation"] * 3)
        expected["label"].extend([taxonomy, taxonomy, ["999", "999", "Unknown", "Unknown", "Unknown", "Unknown", "Unknown"]])
    result = json.loads((tmp_path / "data_index.json").read_text())
    assert result == expected
    contents = [(tmp_path / name).read_bytes() for name in result["path"]]
    build_data_index(str(tmp_path), "train_mini", "val")
    assert json.loads((tmp_path / "data_index.json").read_text()) == expected
    assert [(tmp_path / name).read_bytes() for name in result["path"]] == contents
