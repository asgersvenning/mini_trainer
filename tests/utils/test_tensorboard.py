import numpy as np
import pytest
from mini_trainer.utils.tensorboard import make_empty_array

def test_make_empty_array():
    arr = make_empty_array(5)
    assert arr.shape == (5,)
    assert np.all(np.isnan(arr))
    assert arr.dtype == np.float64
