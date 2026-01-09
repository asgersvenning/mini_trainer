from mini_trainer.utils.parquet import set2split, combine_dicts, get_keys, KCOLUMNS

def test_set2split():
    assert set2split(0) == "test"
    assert set2split(1) == "validation"
    assert set2split(2) == "train"
    assert set2split(99) == "train"

def test_combine_dicts():
    d1 = {"a": 1, "b": 2}
    d2 = {"a": 3, "b": 4}
    combined = combine_dicts([d1, d2])
    assert combined["a"] == [1, 3]
    assert combined["b"] == [2, 4]

def test_get_keys():
    row = {k: str(i) for i, k in enumerate(KCOLUMNS)}
    keys = get_keys(row)
    assert len(keys) == len(KCOLUMNS)
    assert keys[0] == "0"
