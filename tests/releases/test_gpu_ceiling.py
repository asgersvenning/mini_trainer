from types import SimpleNamespace

import pytest
import torch

from dev.releases.mambo_v3.gpu_ceiling import resident_call


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_resident_call_cycles_bank_and_keeps_native_ranks(device):
    if device != "cpu" and not torch.cuda.is_available():
        pytest.skip("Intentional CUDA check")
    bank = torch.arange(8, dtype=torch.uint8, device=device).reshape(8, 1, 1, 1)
    events = []
    seen = []

    def infer(images, embeddings, *, tensors):
        assert tensors and not embeddings and images.device == bank.device
        seen.append(images)
        events.append(object())
        return [images.float().flatten(1) + rank for rank in range(3)], None

    plan = SimpleNamespace(torch_values=lambda leaf, native: (native, None, None))
    predictor = SimpleNamespace(_torch=infer, _model_events=events, selected=None, hierarchy_plan=lambda selected: plan)
    call = resident_call(predictor, bank, 4)
    for i in range(5):
        values = call()
        assert not events
        assert len(values) == 3 and all(v.device == bank.device for v in values)
        expected = bank[(i % 2) * 4 : (i % 2 + 1) * 4]
        torch.testing.assert_close(values[0], expected.float().flatten(1))
        assert seen[-1].data_ptr() == expected.data_ptr()
