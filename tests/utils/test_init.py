import csv
from collections import OrderedDict

import pytest

from mini_trainer.utils import (
    cosine_schedule_with_warmup,
    decimals,
    filter_ordered_dict,
    float_signif_decimal,
    increment_name_dir,
    recursive_dfs_attr,
    write_csv_from_dict,
)


def test_write_csv_from_dict(tmp_path):
    d = {"a": [1, 2], "b": [3, 4]}
    p = tmp_path / "test.csv"
    write_csv_from_dict(d, str(p))

    with open(p) as f:
        reader = csv.reader(f)
        rows = list(reader)
        assert rows[0] == ["a", "b"]
        assert rows[1] == ["1", "3"]
        assert rows[2] == ["2", "4"]

    # Append
    d2 = {"a": [5], "b": [6]}
    write_csv_from_dict(d2, str(p))
    with open(p) as f:
        reader = csv.reader(f)
        rows = list(reader)
        assert len(rows) == 4
        assert rows[3] == ["5", "6"]

    # Mismatch length
    with pytest.raises(ValueError):
        write_csv_from_dict({"a": [1], "b": [1, 2]}, str(p))


def test_filter_ordered_dict():
    od = OrderedDict([("a", 1), ("b", 2), ("c", 3)])
    res = filter_ordered_dict(od, ("a", "c"))
    assert list(res.keys()) == ["a", "c"]
    assert res["a"] == 1
    assert res["c"] == 3


def test_float_signif_decimal():
    assert float_signif_decimal(0.001, digits=3) >= 3
    assert float_signif_decimal(100.0) >= 0
    assert float_signif_decimal(0) == 0


def test_decimals():
    assert decimals(1.234) == 3
    assert decimals(1.0) == 0


def test_increment_name_dir(tmp_path):
    name = "run"
    p = tmp_path

    # 0 -> run
    n1 = increment_name_dir(name, str(p))
    assert n1 == "run"
    (p / "run.txt").touch()

    # 1 -> run_1
    n2 = increment_name_dir(name, str(p))
    assert n2 == "run_1"
    (p / "run_1.txt").touch()

    # 2 -> run_2
    n3 = increment_name_dir(name, str(p))
    assert n3 == "run_2"


def test_recursive_dfs_attr():
    class A:
        def __init__(self):
            self.x = 1

    class B:
        def __init__(self):
            # The function expects iterable objects to traverse
            self.vals = [A(), A()]
            self.x = 99

        def __iter__(self):
            return iter(self.vals)

    b = B()
    assert recursive_dfs_attr(b, "x") == 99

    val = recursive_dfs_attr([A()], "x")
    assert val == 1


def test_cosine_schedule_with_warmup():
    fn = cosine_schedule_with_warmup(total=10, warmup=2, start=0.1, end=0.0)
    assert fn(0) == 0.1
    assert 0.1 < fn(1) < 1.0
    assert fn(2) == 1.0
    for i in range(3, 10):
        assert fn(i + 1) < fn(i)
        assert 0.0 < fn(i) < 1.0
    assert fn(10) == 0.0
