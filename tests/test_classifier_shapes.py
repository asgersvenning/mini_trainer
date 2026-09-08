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
