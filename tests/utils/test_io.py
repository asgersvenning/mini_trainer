import os
import tempfile

import pytest
import torch

from mini_trainer.data.io import (
    CACHE_MODE,
    _normalize_to_tuple,
    generate_indices,
    guess_cache_mode,
    is_image,
    reweight,
)


def test_guess_cache_mode():
    mode = guess_cache_mode([100, 100, 3], torch.uint8)
    assert isinstance(mode, CACHE_MODE)

    thresholds = {
        CACHE_MODE.NONE: -1,
        CACHE_MODE.DISK: 0.0,
        CACHE_MODE.CPU: 0.0,
    }
    assert guess_cache_mode([100, 100, 3], torch.uint8, thresholds=thresholds) == CACHE_MODE.NONE


def test_cache_mode_enum():
    assert CACHE_MODE.NONE == 0
    assert CACHE_MODE(0) == CACHE_MODE.NONE
    assert CACHE_MODE["NONE"] == CACHE_MODE.NONE
    assert CACHE_MODE("none") == CACHE_MODE.NONE
    assert CACHE_MODE(" Disk ") == CACHE_MODE.DISK

    with pytest.raises(ValueError):
        CACHE_MODE("invalid")


def test_is_image():
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        f.write(b"not an image")
        txt_path = f.name

    assert is_image(txt_path) is False
    os.remove(txt_path)

    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(b"\xff\xd8\xff\xe0")
        jpg_path = f.name

    assert is_image(jpg_path) is True
    os.remove(jpg_path)


def test_reweight():
    weights = [1.0, 2.0]
    target_sum = 6.0
    assert reweight(weights, target_sum) == [2.0, 4.0]


def test_generate_indices():
    weights = [1, 2]
    indices, final_weights = generate_indices(weights)
    assert indices == [0, 1, 1]
    assert final_weights == [1, 2]
    # indices -> [0, 0, 1, 1, 1, 1]
    indices, final_weights = generate_indices(weights, target_size=6)
    assert len(indices) == 6
    assert indices.count(0) == 2
    assert indices.count(1) == 4


def test_normalize_to_tuple():
    assert _normalize_to_tuple(1) == (1,)
    assert _normalize_to_tuple([1]) == [1]
    assert _normalize_to_tuple((1,)) == (1,)
