import torch
import pytest
import os
import tempfile
from mini_trainer.utils.io import (
    guess_cache_mode, 
    CACHE_MODE, 
    is_image, 
    reweight, 
    generate_indices, 
    _normalize_to_tuple
)

def test_guess_cache_mode():
    # Small shape, should fit in RAM usually, but depends on thresholds
    # Default thresholds: DISK: 0.5, CPU: 0.5
    # If we pass a shape that is small, memory_proportion should be small.
    # memory_proportion is imported from utils so we can't easily mock it without more work,
    # but we can test logic with custom thresholds
    
    # Mocking memory_proportion might be needed if we want deterministic tests related to system memory.
    # However, we can test that it returns a valid CACHE_MODE.
    
    mode = guess_cache_mode([100, 100, 3], torch.uint8)
    assert isinstance(mode, CACHE_MODE)

    # Test explicit thresholds
    # If we set CPU threshold negative, it should not be picked? 
    # guess_cache_mode logic:
    # if threshold < 0 or memory_proportion(...) < threshold: accepted.append(mode)
    # Default NONE is -1, so it's always accepted if not overridden?
    # Actually NONE threshold is -1.
    # if -1 < 0 -> True. So NONE is always accepted.
    # It returns sorted(accepted)[-1]. NONE is 0. So if anything else is accepted, it wins.
    
    # Force only NONE
    thresholds = {
        CACHE_MODE.NONE: -1,
        CACHE_MODE.DISK: 0.0, # 0.0 threshold means likely fail unless memory usage is 0?
        CACHE_MODE.CPU: 0.0
    }
    # With 0.0 threshold, memory_proportion likely > 0, so not accepted.
    # So should return NONE.
    assert guess_cache_mode([100, 100, 3], torch.uint8, thresholds=thresholds) == CACHE_MODE.NONE

def test_is_image():
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        f.write(b"not an image")
        txt_path = f.name
    
    assert is_image(txt_path) is False
    os.remove(txt_path)
    
    # We could simulate an image header
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(b'\xff\xd8\xff\xe0') # JPEG start
        jpg_path = f.name
    
    assert is_image(jpg_path) is True
    os.remove(jpg_path)

def test_reweight():
    weights = [1.0, 2.0]
    target_sum = 6.0
    # 1+2 = 3. 6/3 = 2.
    # 1*2 = 2, 2*2 = 4. Sum = 6.
    assert reweight(weights, target_sum) == [2.0, 4.0]

def test_generate_indices():
    weights = [1, 2]
    # Default oversample
    indices, final_weights = generate_indices(weights)
    # weights become [1, 2] -> indices [0, 1, 1]
    assert indices == [0, 1, 1]
    assert final_weights == [1, 2]
    
    # With target size
    # target size 6. weights [1, 2] -> sum 3.
    # reweight -> [2, 4]. sum 6.
    # indices -> [0, 0, 1, 1, 1, 1]
    indices, final_weights = generate_indices(weights, target_size=6)
    assert len(indices) == 6
    assert indices.count(0) == 2
    assert indices.count(1) == 4

def test_normalize_to_tuple():
    assert _normalize_to_tuple(1) == (1,)
    assert _normalize_to_tuple([1]) == [1]
    assert _normalize_to_tuple((1,)) == (1,)
