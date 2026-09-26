from mini_trainer.integrations.parquet import KCOLUMNS, combine_dicts, get_keys, set2split


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


def test_get_keys_normalizes_ids_in_taxonomic_order():
    row = {k: f" 00{i} " for i, k in reversed(list(enumerate(KCOLUMNS)))}
    assert get_keys(row) == [str(i) for i in range(len(KCOLUMNS))]
