
from itertools import chain

import torch
from torch import nn as nn

from mini_trainer.classifier import Classifier


def shape_resize(shape : torch.Size | list[int], dim : int, value : int):
    shape = list(shape)
    shape[dim] = value
    return shape

def batched_scatter_logsumexp(input : torch.Tensor, index : torch.Tensor, dim : int=1):
    """
    Aggregates the elements of the ``input`` tensor with an index along a dimension using logsumexp.
    
    ```
    out[j][i] = input[j][index == i].logsumexp()
    ```

    OBS: Behavior for indexes that do not contain all integers from
        0 to :math:`max(index)` or when ``dim`` is not 1 is not defined.

    Args:
        input: Input tensor of size :math:`N x K`.
        index: Long-Tensor of size :math:`K` containing the elements along ``dim`` in ``input`` to aggregate.
        dim: Dimension to aggregate over (default=1).

    Returns:
        output: Aggregated logsumexp of ``input`` of size :math:`N x max(index)+1`.
    """
    z = input.new_zeros(shape_resize(input.shape, dim=dim, value=index.max() + 1)) # Scaffold tensor - same size as output
    index = index.expand_as(input)
    c = z.scatter_reduce(dim=dim, index=index, src=input, reduce="amax", include_self=False)
    return z.scatter_add(dim=dim, index=index, src=(input - c.gather(dim=dim, index=index)).exp()).log() + c

class HierarchicalClassifier(Classifier):
    def __init__(self, sparse_masks : list[torch.Tensor] | None=None, masks : None=None, **kwargs):
        """
        Args:
            sparse_masks: Long-Tensors with parent indices for each element in layers n-1.
            masks: DEPRECATED! Dense child-parent "log-adjacency" matrices.
        """
        super().__init__(**kwargs)

        # Store masks
        self._num_masks = 0
        if sparse_masks is not None:
            [self.register_buffer(f'mask_{i}', m, persistent=True) for i, m in enumerate(sparse_masks)]
            self._num_masks = len(sparse_masks)

    @property
    def num_masks(self):
        stored = getattr(self, "_num_masks", None)
        if not isinstance(stored, int):
            i = 0
            while hasattr(self, f'mask_{i}'):
                i += 1
            self._num_masks = stored = i
        return stored

    def mask(self, idx : int) -> torch.Tensor:
        return getattr(self, f'mask_{idx}')

    @property
    def masks(self):
        return [self.mask(i) for i in range(self.num_masks)]

    def hierarchy(self, log_probs : torch.Tensor):
        ys = [log_probs]
        # Propagate the probabilities up the hierarchy using the masks
        for mask in self.masks:
            ys.append(batched_scatter_logsumexp(ys[-1], mask))
        return ys
    
    def forward(self, x):
        return self.hierarchy(super().forward(x).log_softmax(dim=1))


class ConditionalClassifier(HierarchicalClassifier):
    def __init__(self, in_features : int, normalized : bool=True, **kwargs):
        super().__init__(in_features=in_features, normalized=normalized, **kwargs)
        # Conditional layers
        layers : list[nn.Linear] = []
        for m in self.masks:
            out = int(m.max().item() + 1)
            layer = nn.Linear(in_features, out, bias=True)
            layers.append(self._normalize_layer(layer) if normalized else layer)
        self.layers = nn.ModuleList(layers)

    @property
    def last_layers(self):
        return chain([self.linear], self.layers)

    def marginals(self, x : torch.Tensor) -> list[torch.Tensor]:
        return [layer(x).log_softmax(dim=1) for layer in self.last_layers]

    def forward(self, x : torch.Tensor):
        M = self.marginals(self.preclassification(x))
        C : list[torch.Tensor] = []
        for i in reversed(range(len(M))):
            if not C:
                ci = M[i]
            else:
                mask = self.mask(i)
                ci = M[i] + (C[0] - batched_scatter_logsumexp(M[i], mask)).gather(1, mask.expand_as(M[i]))
            C.insert(0, ci)
        return C


class IndependentClassifier(ConditionalClassifier):
    def forward(self, x):
        return super().marginals(super().preclassification(x))