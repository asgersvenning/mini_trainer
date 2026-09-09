import pytest
import torch
import torch.nn.functional as F

from mini_trainer.modeling import Classifier
from mini_trainer.utils import cosine_to_zscore


@pytest.mark.parametrize("shape", [(2, 8), (2, 1, 8), (2, 8, 1), (2, 8, 1, 1), (2, 1, 1, 8)])
@pytest.mark.parametrize("hidden", [False, True, 4, 12])
@pytest.mark.parametrize("normalized", [False, True])
def test_backbone_width_and_embedding_width(shape, hidden, normalized):
    head = Classifier(in_features=8, out_features=3, hidden=hidden, normalized=normalized).eval()
    inputs = torch.randn(2, 8)
    embedding_width = 8 if isinstance(hidden, bool) else hidden
    with torch.no_grad():
        embeddings = F.leaky_relu(head.hidden(inputs)) if hidden else inputs
        embeddings = F.normalize(embeddings, dim=-1) if normalized else head.batch_norm(embeddings)
        expected = F.linear(embeddings, head.linear.weight)
        if normalized:
            expected = cosine_to_zscore(expected, embedding_width)
        expected = expected + head.linear.bias
        actual = head(inputs.reshape(shape))
    assert head.in_features == 8
    assert head.preclassification_size == embedding_width
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("normalized", [False, True])
def test_custom_hidden_width_checkpoint_roundtrip(normalized):
    head = Classifier(in_features=8, out_features=3, hidden=4, normalized=normalized).eval()
    state = head.state_dict()
    restored = Classifier(**state["_extra_state"]).eval()
    restored.load_state_dict(state, strict=True)
    inputs = torch.randn(2, 8)
    with torch.no_grad():
        torch.testing.assert_close(restored(inputs), head(inputs))


@pytest.mark.parametrize("shape", [(8, 32), (32, 8)])
def test_spherical_initialization_preserves_reference_update(shape):
    from mini_trainer.modeling import Classifier

    layer = torch.nn.Linear(shape[1], shape[0], bias=False)
    torch.manual_seed(73)
    reference = torch.empty_like(layer.weight).normal_()
    for _ in range(100):
        reference.div_(reference.norm(dim=1, keepdim=True).clamp(min=1e-9))
        gradient = reference @ reference.t() @ reference
        projection = (gradient * reference).sum(dim=1, keepdim=True) * reference
        reference.sub_((0.5 / shape[0]) * (gradient - projection))
    reference.div_(reference.norm(dim=1, keepdim=True).clamp(min=1e-9))
    torch.manual_seed(73)
    result = Classifier.init_spherical_repulsion(layer)
    assert result is layer
    torch.testing.assert_close(layer.weight, reference, rtol=2e-5, atol=2e-6)


def test_large_normalized_head_does_not_allocate_class_gram_matrix():
    from torch.utils._python_dispatch import TorchDispatchMode

    from mini_trainer.modeling import Classifier

    class RejectClassGram(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func == torch.ops.aten.mm.default:
                left, right = args
                assert (left.shape[0], right.shape[1]) != (10000, 10000)
            return func(*args, **(kwargs or {}))

    with RejectClassGram():
        head = Classifier(in_features=8, out_features=10000, hidden=True, normalized=True)
    assert torch.isfinite(head.linear.weight).all()
    torch.testing.assert_close(head.linear.weight.norm(dim=1), torch.ones(10000))
