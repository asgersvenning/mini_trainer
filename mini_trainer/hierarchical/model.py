
from dataclasses import dataclass
from functools import lru_cache
from itertools import chain

import torch
from torch import nn as nn

from mini_trainer.classifier import Classifier, Prediction


def shape_resize(shape : torch.Size | list[int], dim : int, value : int): # noqa: D103
    shape = list(shape)
    shape[dim] = value
    return shape


def batched_scatter_logsumexp(input : torch.Tensor, index : torch.Tensor, dim : int=1):
    """Aggregates the elements of the ``input`` tensor with an index along a dimension using logsumexp.
    
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
    # Scaffold tensor - same size as output
    z = input.new_zeros(shape_resize(input.shape, dim=dim, value=index.max() + 1))
    index = index.expand_as(input)
    c = z.scatter_reduce(dim=dim, index=index, src=input, reduce="amax", include_self=False)
    return z.scatter_add(dim=dim, index=index, src=(input - c.gather(dim=dim, index=index)).exp()).log() + c


class HierarchicalClassifier(Classifier): # noqa: D101 TODO
    def __init__(self, sparse_masks : list[torch.Tensor] | None=None, masks : None=None, **kwargs):
        """TODO.

        Args:
            sparse_masks: Long-Tensors with parent indices for each element in layers n-1.
            masks: DEPRECATED! Dense child-parent "log-adjacency" matrices.
            kwargs: passed to `mini_trainer.classifier.Classifier`.
        """
        super().__init__(**kwargs)

        # Store masks
        self._num_masks = 0
        if sparse_masks is not None:
            [self.register_buffer(f'mask_{i}', m.long(), persistent=True) for i, m in enumerate(sparse_masks)]
            self._num_masks = len(sparse_masks)

    @classmethod
    def load(cls, architecture_class, architecture_output_name, architecture, state, device, dtype, **kwargs):
        if state is not None:
            prefix = f"{architecture_output_name}.mask_"
            # Extract (index, tensor) tuples, sort by index, and retrieve tensors
            masks = [(int(k.split("_")[-1]), v) for k, v in state.items() if k.startswith(prefix)]
            if masks:
                kwargs["sparse_masks"] = [v for _, v in sorted(masks)]

        return super().load(
            architecture_class, architecture_output_name, architecture, state, device, dtype, **kwargs
        )

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
        masks = []
        filter = self.active_indices
        filters = [filter]
        for i in range(self.num_masks):
            mask = self.mask(i)
            if filter is not None:
                mask = mask[filter]
                filter, mask = mask.unique(sorted=False, return_inverse=True)
            masks.append(mask)
            filters.append(filter)
        self._active_indices = filters
        return masks

    def hierarchy(self, log_probs : torch.Tensor):
        ys = [log_probs]
        # Propagate the probabilities up the hierarchy using the masks
        for mask in self.masks:
            ys.append(batched_scatter_logsumexp(ys[-1], mask))
        return ys
    
    def forward(self, x):
        return self.hierarchy(super().forward(x).log_softmax(dim=1))
    
    @torch.no_grad()
    def predict(self, x, topk : int=1, **kwargs):
        out = list(zip(*self(x)))
        return [
            HierarchicalPrediction(p, topk=topk, active_indices=self._active_indices, **{**self._metadata, **kwargs}) 
            for p in out
        ]


class ConditionalClassifier(HierarchicalClassifier): # noqa: D101 TODO
    def __init__(self, normalized : bool=True, **kwargs): # noqa: D107
        super().__init__(normalized=normalized, **kwargs)
        # Conditional layers
        layers : list[nn.Linear] = []
        orig_indices = self.active_indices
        self.active_indices = None
        for m in self.masks:
            out = int(m.max().item() + 1)
            layer = nn.Linear(self.preclassification_size, out, bias=True)
            layers.append(self._normalize_layer(layer) if normalized else layer)
        self.active_indices = orig_indices
        self.layers = nn.ModuleList(layers)

    @property
    def last_layers(self):
        return chain([self.linear], self.layers)

    # def marginals(self, x : torch.Tensor) -> list[torch.Tensor]:
    #     return [layer(x).log_softmax(dim=1) for layer in self.last_layers]
    def marginals(self, x: torch.Tensor) -> list[torch.Tensor]:
        outputs = [self.linear(x).log_softmax(dim=1)]

        indices = self.active_indices
        for i, layer in enumerate(self.layers):
            if indices is not None:
                indices = self.mask(i)[indices].unique()
            out = layer(x)
            if indices is not None:
                out = out.index_select(1, indices)
            outputs.append(out.log_softmax(dim=1))
            
        return outputs

    def forward(self, x : torch.Tensor):
        M = self.marginals(self.preclassification(x))
        C : list[torch.Tensor] = []
        masks = self.masks
        for i in reversed(range(len(M))):
            if not C:
                ci = M[i]
            else:
                ci = M[i] + (C[0] - batched_scatter_logsumexp(M[i], masks[i])).gather(1, masks[i].expand_as(M[i]))
            C.insert(0, ci)
        return C


class IndependentClassifier(ConditionalClassifier): # noqa: D101 TODO
    def forward(self, x):
        return super().marginals(super().preclassification(x))


@dataclass
class HierarchicalPredictionItem:
    """Hierarchical data container.
    Auto-converts inputs to native Python types on initialization.
    """
    label: tuple[str, ...] | None
    confidence: tuple[float, ...]
    index: tuple[int, ...] | None

    def __post_init__(self):
        # Factory coercion: ensures native types immediately
        if self.label is not None:
            self.label = tuple(str(e) for e in self.label)
        if self.confidence is not None:
            self.confidence = tuple(float(e) for e in self.confidence)
        if self.index is not None:
            self.index = tuple(int(e) for e in self.index)

    def __repr__(self):
        if self.label is not None:
            return "/".join([f'{label}: ({conf:.1%})' for conf, label in zip(self.confidence, self.label)])
        else:
            return "/".join([f'I[{idx}]: ({conf:.1%})' for conf, idx in zip(self.confidence, self.index)])

    def to_dict(self):
        return {
            "label": self.label,
            "confidence": self.confidence,
            "index": self.index
        }


class HierarchicalPrediction(Prediction):
    """Hierarchical PyTorch Prediction class.
    """
    ITEM_CLASS = HierarchicalPredictionItem

    def __init__(
            self, 
            *args, 
            cls2idx : dict[str, dict[str, int]], 
            active_indices : list[list[int] | torch.Tensor | None] | None=None, 
            **kwargs
        ):
        if active_indices is not None and not any(ai is None for ai in active_indices):
            active_indices = [sorted(ai.tolist() if isinstance(ai, torch.Tensor) else ai) for ai in active_indices]
            reindex = [{old : new for new, old in enumerate(ai)} for ai in active_indices]
            cls2idx = {
                outer : {k : reindex[i][v] for k, v in inner.items() if v in reindex[i]}
                for i, (outer, inner) in enumerate(cls2idx.items())
            }
        super().__init__(*args, cls2idx=cls2idx, **kwargs)

    @property
    @lru_cache(1)
    def idx2cls(self):
        if self.cls2idx:
            return {level : {v: k for k, v in mapping.items()} for level, mapping in self.cls2idx.items()}
        return None

    def _process(self, raw_prediction : list[torch.Tensor]):
        logits, indices = [], []
        for rp in raw_prediction:
            dim = rp.shape[-1]
            k = self.topk if 1 <= self.topk < dim else dim
            lgs, idx = torch.topk(rp, k)
            logits.append(lgs)
            indices.append(idx)
        return [torch.stack(v) for v in zip(*logits)], [torch.stack(v) for v in zip(*indices)]

    def _translate(self) -> list[list[str]]:
        if self.idx2cls:
            return [[self.idx2cls[str(level)][i.item()] for level, i in enumerate(e)] for e in self.indices]
        else:
            return [[f'{i.item()}[{level}]' for level, i in enumerate(e)] for e in self.indices]

    def _extract_confidence(self, raw_prediction: list[torch.Tensor]) -> list[torch.Tensor]:
        confidences = []
        for rp, idx in zip(raw_prediction, zip(*self.indices)):
            idx = torch.stack(idx)
            if not (
                torch.all(rp >= 0) and
                torch.isclose(
                    rp.sum(), 
                    torch.tensor(1.0, dtype=rp.dtype), 
                    atol=1e-3 if rp.dtype == torch.float16 else 1e-5
                )
            ):
                rp = rp.softmax(dim=-1)
            confidences.append(rp[idx])
        return [torch.stack(v) for v in zip(*confidences)]