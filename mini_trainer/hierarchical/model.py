
from dataclasses import dataclass
from functools import lru_cache

import torch
from torch import nn as nn
from torch.nn import functional as F

from mini_trainer.classifier import BasePrediction, Classifier, PredictionItem, SupervisionContext


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
            for i, m in enumerate(sparse_masks):
                self.register_buffer(f'mask_{i}', m.long(), persistent=True)
                self.register_buffer(f'_mask_{i}', torch.empty(0), persistent=False)
                self.register_buffer(f'_filter_{i}', torch.empty(0), persistent=False)
            self._num_masks = len(sparse_masks)
            self.register_buffer(f'_filter_{self.num_masks}', torch.empty(0), persistent=False)

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
        if self._dirty_cache["_masks"]:
            _ = self.masks
        return getattr(self, f'_mask_{idx}')

    @property
    def masks(self):
        if self._dirty_cache["_masks"]:
            masks = []
            filter = self.active_indices
            filters = [filter]
            for i in range(self.num_masks):
                mask = getattr(self, f"mask_{i}")
                setattr(self, f"_filter_{i}", filter)
                if filter is not None:
                    mask = mask[filter]
                    filter, mask = mask.unique(sorted=False, return_inverse=True)
                setattr(self, f"_mask_{i}", mask)
                masks.append(mask)
                filters.append(filter)
            setattr(self, f"_filter_{self.num_masks}", filter)
            self._dirty_cache["_masks"] = False
        masks : list[torch.Tensor] = []
        for i in range(self.num_masks):
            masks.append(self.mask(i))
        return masks

    def hierarchy(self, log_probs : torch.Tensor):
        ys = [log_probs]
        # Propagate the probabilities up the hierarchy using the masks
        for mask in self.masks:
            ys.append(batched_scatter_logsumexp(ys[-1], mask))
        return ys
    
    def forward(self, x):
        return self.hierarchy(super().forward(x).log_softmax(dim=1))
    
    def _preprocess_metadata(self, cls2idx=None, **kwargs):
        if self._dirty_cache["_preprocess_metadata"] or cls2idx is not None:
            if self._dirty_cache["_masks"]:
                _ = self.masks
            active_indices = [getattr(self, f"_filter_{i}") for i in range(self.num_masks + 1)]
            metadata = self._metadata.copy()
            if cls2idx is None:
                cls2idx = metadata.get("cls2idx", None)
            if active_indices is not None and not any(ai is None for ai in active_indices) and cls2idx is not None:
                active_indices = [sorted(ai.tolist() if isinstance(ai, torch.Tensor) else ai) for ai in active_indices]
                reindex = [{old : new for new, old in enumerate(ai)} for ai in active_indices]
                cls2idx = {
                    outer : {k : reindex[i][v] for k, v in inner.items() if v in reindex[i]}
                    for i, (outer, inner) in enumerate(cls2idx.items())
                }
                metadata["cls2idx"] = cls2idx
            self._preprocessed_metadata = metadata
            self._dirty_cache["_preprocess_metadata"] = False
        return {**self._preprocessed_metadata, **kwargs}

    @torch.no_grad()
    def predict(self, x, topk : int=1, **kwargs):
        out = list(zip(*self(x)))
        return [
            HierarchicalPrediction(p, topk=topk, **self._preprocess_metadata(**kwargs)) 
            for p in out
        ]


class ConditionalClassifier(HierarchicalClassifier): # noqa: D101 TODO
    def __init__(self, normalized : bool=True, **kwargs): # noqa: D107
        super().__init__(normalized=normalized, **kwargs)
        # Conditional layers
        layers : list[nn.Linear] = [self.linear]
        orig_indices = self.active_indices
        self.active_indices = None
        for m in self.masks:
            out = int(m.max().item() + 1)
            layer = nn.Linear(self.preclassification_size, out, bias=True)
            layers.append(self._normalize_layer(layer) if normalized else layer)
        self.active_indices = orig_indices
        self.layers = nn.ModuleList(layers)
        for i in range(len(self.layers)):
            self.register_buffer(f"_linear_weight_{i}", torch.empty(0), persistent=False)
            self.register_buffer(f"_linear_bias_{i}", torch.empty(0), persistent=False)

    def _weight_bias(self, i : int):
        if self._dirty_cache[f"_weight_bias_{i}"] or self.training:
            if self._dirty_cache["_masks"]:
                _ = self.masks
            layer : nn.Linear = self.layers[i]
            weight, bias = layer.weight, layer.bias
            filter = getattr(self, f"_filter_{i}")
            if filter is not None:
                weight = weight.index_select(0, filter)
                if bias is not None:
                    bias = bias.index_select(0, filter)
            setattr(self, f"_linear_weight_{i}", weight)
            setattr(self, f"_linear_bias_{i}", bias)
            self._dirty_cache[f"_weight_bias_{i}"] = False
        return getattr(self, f"_linear_weight_{i}"), getattr(self, f"_linear_bias_{i}")

    def marginals(self, x : torch.Tensor) -> list[torch.Tensor]:
        return [F.linear(x, *self._weight_bias(i)).log_softmax(dim=1) for i in range(len(self.layers))]

    def forward(self, x : torch.Tensor):
        M = self.marginals(self.preclassification(x))
        N = len(M)
        C : list[torch.Tensor] = [torch.empty(0) for _ in range(N)]
        for i in reversed(range(N)):
            if i < N - 1:
                M[i] += (C[i + 1] - batched_scatter_logsumexp(M[i], self.mask(i))).gather(1, self.mask(i).expand_as(M[i]))
            C[i] = M[i]
        return C


class IndependentClassifier(ConditionalClassifier): # noqa: D101 TODO
    def forward(self, x):
        return super().marginals(super().preclassification(x))


class AutoregressiveClassifier(IndependentClassifier): # noqa: D101 TODO
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sequence_length = len(self.layers) + 1 # total = layers + <BOS>
        self.positional = nn.Embedding(
            num_embeddings=self.sequence_length,
            embedding_dim=self.preclassification_size
        )
        self.BOS = nn.Embedding(
            num_embeddings=1,
            embedding_dim=self.preclassification_size
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=self.preclassification_size,
                nhead=8, # TODO: Should not be hardcoded
                dim_feedforward=self.preclassification_size,
                dropout=0.1,
                norm_first=True
            ),
            num_layers=2 # TODO: Should not be hardcoded
        )

    def embedding(self, i : int) -> torch.Tensor:
        return self._weight_bias(i)[0]
    
    @property
    def embeddings(self):
        return [self.embedding(i) for i in range(len(self.layers))]

    def _classify(self, sequence : torch.Tensor | list[torch.Tensor]) -> list[torch.Tensor]:
        if isinstance(sequence, torch.Tensor):
            sequence = [e for e in sequence]        
        return [
            (x @ self.embedding(i).T).log_softmax(dim=1) 
            for i, x in list(enumerate(sequence[::-1]))
        ]

    def forward(self, x : torch.Tensor, y : torch.Tensor | list[int] | None=None):
        if y is None:
            # Check if a label is passed around parent module via context manager
            y = SupervisionContext.get()

        # Image embedding context
        context = self.preclassification(x)
        
        # Prepare variables and state
        batch_size, _, device = context.shape[0], context.dtype, context.device
        mask = nn.Transformer.generate_square_subsequent_mask(self.sequence_length, device=device)
        BOS : torch.Tensor = self.BOS(torch.zeros((batch_size,), dtype=torch.long, device=device))
        POS = self.positional.weight.unsqueeze(1) 

        if y is None or torch.rand((1, )).item() > 0.5:
            decision = BOS.unsqueeze(0).repeat(self.sequence_length, 1, 1)
            for i in range(self.sequence_length - 1):
                sequence = self.decoder(
                    tgt=decision + POS, 
                    memory=context.unsqueeze(0), 
                    tgt_mask=mask, 
                    tgt_is_causal=True
                )
                di = self.sequence_length - 2 - i
                decision[i + 1] = (sequence[i] @ self.embedding(di).T).softmax(dim=1) @ self.embedding(di)
        else:
            sequence = [self.embedding(j)[y[:, j]] for j in range(self.sequence_length - 1)]
            sequence.append(BOS)
            sequence = torch.stack(sequence[::-1], dim=0)
            sequence = self.decoder(
                tgt=sequence + POS,
                memory=context.unsqueeze(0),
                tgt_mask=mask,
                tgt_is_causal=True
            )
        
        return self._classify(sequence[:-1])


# Differs from the one above in that we don't carry explicit independent embeddings for each layer
class AutoregressiveClassifierV2(HierarchicalClassifier): # noqa: D101 TODO
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sequence_length = self.num_masks + 1 + 1 # total = masks + leaf + <BOS>
        self.positional = nn.Embedding(
            num_embeddings=self.sequence_length,
            embedding_dim=self.preclassification_size
        )
        self.BOS = nn.Embedding(
            num_embeddings=1,
            embedding_dim=self.preclassification_size
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=self.preclassification_size,
                nhead=8, # TODO: Should not be hardcoded
                dim_feedforward=self.preclassification_size,
                dropout=0.1,
                norm_first=True
            ),
            num_layers=2 # TODO: Should not be hardcoded
        )
    
    @property
    def embeddings(self):
        return self._weight_bias()[0]

    def _classify(self, sequence : torch.Tensor | list[torch.Tensor]) -> list[torch.Tensor]:
        if isinstance(sequence, torch.Tensor):
            sequence = [e for e in sequence]        
        return [
            self.hierarchy(x @ self.embeddings.T)[i].log_softmax(dim=1)
            for i, x in list(enumerate(sequence[::-1]))
        ]

    def forward(self, x : torch.Tensor, y : torch.Tensor | list[int] | None=None):
        if y is None:
            # Check if a label is passed around parent module via context manager
            y = SupervisionContext.get()

        # Image embedding context
        context = self.preclassification(x)
        
        # Prepare variables and state
        batch_size, dtype, device = context.shape[0], context.dtype, context.device
        mask = nn.Transformer.generate_square_subsequent_mask(self.sequence_length, device=device)
        BOS : torch.Tensor = self.BOS(torch.zeros((batch_size,), dtype=torch.long, device=device))
        POS = self.positional.weight.unsqueeze(1) 

        if y is None or torch.rand((1, )).item() > 0.5:
            decision = BOS.unsqueeze(0).repeat(self.sequence_length, 1, 1)
            for i in range(self.sequence_length - 1):
                sequence : torch.Tensor = self.decoder(
                    tgt=decision + POS, 
                    memory=context.unsqueeze(0), 
                    tgt_mask=mask, 
                    tgt_is_causal=True
                )
                logits = (sequence[i] @ self.embeddings.T).log_softmax(dim=1)
                mi = self.num_masks - 1 - i
                if mi > 0:
                    logits = self.hierarchy(logits)[mi]
                    con = self.mask(mi)
                    for mj in range(mi + 1):
                        con = con[self.mask(mi - mj)]
                    logits = (
                        logits.gather(1, con.unsqueeze(0).expand(logits.size(0), -1)) - 
                        con.bincount(minlength=logits.size(1)).to(dtype).log()[con]
                    )                    
                decision[i + 1] = logits.exp() @ self.embeddings
        else:
            sequence : list[torch.Tensor] = []
            for i in range(self.sequence_length - 1):
                if i == 0:
                    emb = self.embeddings[y[:, 0]]
                else:
                    idx = y[:, i]
                    con = self.mask(0)
                    for j in range(1, i):
                        con = self.mask(j)[con]
                    a, b = torch.nonzero((idx == con.unsqueeze(1)).T, as_tuple=True)
                    lidx = b.tensor_split((a.diff() != 0).nonzero(as_tuple=True)[0].cpu() + 1)
                    dist = torch.zeros((batch_size, len(self.mask(0))), device=device, dtype=dtype).requires_grad_(False)
                    for j, k in enumerate(lidx):
                        dist[j][k] = 1 / len(k)
                    emb = dist @ self.embeddings
                sequence.append(emb)
            sequence.append(BOS)
            sequence = torch.stack(sequence[::-1], dim=0)
            sequence : torch.Tensor = self.decoder(
                tgt=sequence + POS,
                memory=context.unsqueeze(0),
                tgt_mask=mask,
                tgt_is_causal=True
            )
        
        return self._classify(sequence[:-1])


@dataclass
class HierarchicalPredictionItem(PredictionItem):
    """Hierarchical data container.
    Auto-converts inputs to native Python types on initialization.
    """
    label: tuple[str, ...]
    confidence: tuple[float, ...]
    index: tuple[int, ...]

    def __post_init__(self):
        # Factory coercion: ensures native types immediately
        self.label = tuple(str(e) for e in self.label)
        self.confidence = tuple(float(e) for e in self.confidence)
        self.index = tuple(int(e) for e in self.index)

    def __repr__(self):
        if self.label is not None:
            return "/".join([f'{label}: ({conf:.1%})' for conf, label in zip(self.confidence, self.label)])
        else:
            return "/".join([f'I[{idx}]: ({conf:.1%})' for conf, idx in zip(self.confidence, self.index)])


class HierarchicalPrediction(BasePrediction[HierarchicalPredictionItem, list[torch.Tensor]]):
    """Hierarchical PyTorch Prediction class.
    """
    ITEM_CLASS = HierarchicalPredictionItem

    @property
    @lru_cache(1)
    def idx2cls(self):
        if self.cls2idx:
            return {level : {v: k for k, v in mapping.items()} for level, mapping in self.cls2idx.items()}
        return None

    def _process(self, raw_prediction):
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

    def _extract_confidence(self, raw_prediction) -> list[torch.Tensor]:
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