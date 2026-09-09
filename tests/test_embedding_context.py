"""Embedding publication must retain gradients without splitting model graphs."""

import copy

import pytest
import torch
from torch._dynamo.testing import CompileCounterWithBackend

from mini_trainer.modeling import Classifier, EmbeddingContext


@pytest.mark.parametrize("normalized", [False, True])
def test_fullgraph_embedding_publication_matches_eager_gradients(normalized):
    torch.manual_seed(27)
    reference = Classifier(16, 4, hidden=8, droprate=0, normalized=normalized)
    model = copy.deepcopy(reference)
    counter = CompileCounterWithBackend("aot_eager")
    compiled = torch.compile(model, backend=counter, fullgraph=True)
    warmed_frames = None
    for iteration in range(4):
        data = torch.randn(8, 16)
        results = []
        for call, parameters in ((reference, reference.parameters()), (compiled, model.parameters())):
            call.zero_grad(set_to_none=True)
            inputs = data.clone().requires_grad_()
            with EmbeddingContext():
                scores = call(inputs)
                embeddings = EmbeddingContext.get()
                assert EmbeddingContext.active() and embeddings is not None and embeddings.requires_grad
                (scores.square().mean() + embeddings[:, 0].sum()).backward()
                results.append(
                    (scores.detach().clone(), inputs.grad.clone(), [None if p.grad is None else p.grad.clone() for p in parameters])
                )
            assert not EmbeddingContext.active() and EmbeddingContext.get() is None
        torch.testing.assert_close(results[0], results[1])
        # The classifier initializes its lazy weight-cache metadata on the
        # first forward. Once that guard settles, publishing fresh embeddings
        # must neither split the graph (fullgraph=True) nor recompile it.
        if iteration == 1:
            warmed_frames = counter.frame_count
        elif iteration > 1:
            assert counter.frame_count == warmed_frames


def test_embedding_context_rejects_nesting_and_cleans_up_after_errors():
    embeddings = torch.randn(2, 3, requires_grad=True)
    with pytest.raises(ValueError, match="body failure"):
        with EmbeddingContext():
            EmbeddingContext.set(embeddings)
            assert EmbeddingContext.get() is embeddings
            with pytest.raises(RuntimeError, match="already active"):
                with EmbeddingContext():
                    pass
            assert EmbeddingContext.active() and EmbeddingContext.get() is embeddings
            raise ValueError("body failure")
    assert not EmbeddingContext.active() and EmbeddingContext.get() is None
