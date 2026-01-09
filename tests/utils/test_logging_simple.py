import time
import pytest
import torch
import numpy as np
from mini_trainer.utils.logging import (
    format_duration,
    Timer,
    ETA,
    accuracy,
    compute_aligned_steps,
    SmoothedValue
)

def test_format_duration():
    assert format_duration(3661) == "01h01m01s"
    assert format_duration(60) == "01m00s"

def test_Timer():
    t = Timer()
    assert not t.running
    t.start()
    assert t.running
    time.sleep(0.01)
    t.stop()
    assert not t.running
    assert t.total >= 0.0

def test_ETA():
    eta = ETA(total_steps=10)
    assert eta.remaining == 10
    eta.step()
    assert eta.remaining == 9
    assert eta.eta is not None

def test_accuracy():
    output = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
    target = torch.tensor([1, 0])
    acc1 = accuracy(output, target, topk=(1,))
    assert acc1[0] == 100.0
    
    output = torch.tensor([[0.9, 0.1], [0.2, 0.8]]) # Wrong
    acc1 = accuracy(output, target, topk=(1,))
    assert acc1[0] == 0.0

def test_compute_aligned_steps():
    # target len 10, origin len 10
    steps = compute_aligned_steps(10, 10, 1, 0)
    assert len(steps) == 10
    assert steps[0] == 0
    assert steps[-1] == 9
    
    # Validation usually has fewer steps or different freq
    # Origin 5, Target 10
    steps = compute_aligned_steps(10, 5, 1, 0)
    assert len(steps) == 5
    # linspace(0, 9, 5) -> 0, 2.25, 4.5, 6.75, 9
    # round: 0, 2, 4 (4.5 rounds to nearest even? or Up? Py3 round ties to even: 4).
    # 6.75 -> 7.
    # 9 -> 9.
    # [0, 2, 4, 7, 9]
    assert steps == [0, 2, 4, 7, 9]

def test_SmoothedValue():
    sv = SmoothedValue(window_size=2)
    sv.update(1.0)
    sv.update(2.0)
    assert sv.count == 2
    assert sv.total == 3.0
    assert sv.value == 2.0
    assert sv.avg == 1.5
    assert sv.global_avg == 1.5
    
    sv.update(3.0) # deque [2.0, 3.0]
    assert sv.value == 3.0
    assert sv.avg == 2.5
    # Global avg is over all history
    assert sv.global_avg == 6.0/3.0 # 2.0
