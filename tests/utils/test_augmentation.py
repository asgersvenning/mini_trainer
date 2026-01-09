import torch
import pytest
from mini_trainer.utils.augmentation import iinfo_maxval_static, salt_and_pepper, SaltAndPepper

def test_iinfo_maxval_static():
    assert iinfo_maxval_static(torch.uint8) == 255
    assert iinfo_maxval_static(torch.int8) == 127
    assert iinfo_maxval_static(torch.int16) == 32767
    
    with pytest.raises((ValueError, torch.jit.Error)):
        iinfo_maxval_static(torch.float32)

def test_salt_and_pepper():
    # Create a small image
    img = torch.zeros((3, 10, 10), dtype=torch.float32)
    
    # Apply salt and pepper with high probability and proportion range
    # probability=1 means always apply
    # proportion=(0.5, 0.5) means 50% of pixels will be modified
    aug_img = salt_and_pepper(img, proportion=(0.5, 0.5), probability=1.0)
    
    # Check that some pixels have changed
    assert not torch.allclose(img, aug_img)
    
    # Check that pixels are either 0 or 1 (since it's float)
    # The original was 0. Salt (1) and Pepper (0) are adding noise.
    # Actually, salt_and_pepper sets pixels to 0 or MAX.
    # For float, MAX is 1.
    
    unique_vals = torch.unique(aug_img)
    # Ideally should be only 0.0 and 1.0, but original image was 0.0 so we might just see 0 and 1.
    for val in unique_vals:
        assert val.item() in [0.0, 1.0]

def test_salt_and_pepper_module():
    mod = SaltAndPepper(proportion=(0.5, 0.5), probability=1.0)
    img = torch.zeros((3, 10, 10), dtype=torch.float32)
    aug_img = mod(img)
    assert not torch.allclose(img, aug_img)
    assert "SaltAndPepper" in repr(mod)
