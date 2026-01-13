import pytest
import torch
from mini_trainer.config import (
    _nullify,
    _drop_none,
    _stringify_types,
    defaults_from_function,
    merge_dicts,
    restructure_cli_args,
)

def test_nullify():
    d = {"a": 1, "b": {"c": 2, "d": {"e": 3}}}
    expected = {"a": None, "b": {"c": None, "d": {"e": None}}}
    assert _nullify(d) == expected

def test_drop_none():
    d = {"a": 1, "b": None, "c": {"d": 2, "e": None}, "f": {}}
    expected = {"a": 1, "c": {"d": 2}}
    assert _drop_none(d) == expected

def test_stringify_types():
    class MyClass:
        pass

    obj = {
        "device": torch.device("cpu"),
        "dtype": torch.float32,
        "type": MyClass,
        "list": [1, 2],
        "tuple": (3, 4),
        "set": {5, 6},
        "str": "test",
        "int": 1,
        "float": 1.0,
        "bool": True,
        "none": None,
        "object": MyClass(),
    }
    
    # Note: Sets are unordered, so convert to sorted list for comparison if needed, 
    # but _stringify_types returns a set for a set input (if it's not converted recursively).
    # Wait, looking at code: 
    # if isinstance(obj, (list, tuple, set)):
    #     seq = [_stringify_types(v) for v in obj]
    #     return type(obj)(seq) if not isinstance(obj, set) else seq
    # So a set returns a list? "else seq". Yes.
    
    result = _stringify_types(obj)
    
    assert result["device"] == "cpu"
    assert result["dtype"] == "float32"
    assert "test_stringify_types.<locals>.MyClass" in result["type"]
    assert result["list"] == [1, 2]
    assert result["tuple"] == (3, 4)
    # The set is converted to a list
    assert sorted(result["set"]) == [5, 6] 
    assert result["str"] == "test"
    assert result["int"] == 1
    assert result["float"] == 1.0
    assert result["bool"] is True
    assert result["none"] is None
    assert "test_stringify_types.<locals>.MyClass" in result["object"]

def test_defaults_from_function():
    def func(a=1, b="test", c=None, d=[1, 2]):
        pass

    defaults = defaults_from_function(func)
    assert defaults["a"] == 1
    assert defaults["b"] == "test"
    assert defaults["c"] is None
    assert defaults["d"] == [1, 2]
    
    # Verify deepcopy
    defaults["d"].append(3)
    assert defaults_from_function(func)["d"] == [1, 2]

def test_merge_dicts():
    d1 = {"a": 1, "b": {"c": 2}}
    d2 = {"b": {"d": 3}, "e": 4}
    merged = merge_dicts(d1, d2)
    assert merged == {"a": 1, "b": {"c": 2, "d": 3}, "e": 4}

    # Later dicts win
    d3 = {"a": 2}
    merged = merge_dicts(d1, d3)
    assert merged["a"] == 2

def test_restructure_cli_args():
    args = {
        "a.b.c": 1,
        "a.b.d": 2,
        "e": 3,
        "f": None,
    }
    expected = {
        "a": {"b": {"c": 1, "d": 2}},
        "e": 3,
    }
    assert restructure_cli_args(args) == expected
