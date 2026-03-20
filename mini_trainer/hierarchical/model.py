from dataclasses import dataclass
from functools import lru_cache

import torch
from torch import nn as nn
from torch.nn import functional as F

from mini_trainer.classifier import BasePrediction, Classifier, EmbeddingContext, PredictionItem, SupervisionContext
from mini_trainer.hierarchical.utils import batched_scatter_logsumexp, prior_from_labels
from mini_trainer.utils.generic import cosine_to_zscore


class HierarchicalClassifier(Classifier): # noqa: D101 TODO
    def __init__(  # noqa: D417
            self, 
            sparse_masks : list[torch.Tensor] | None=None, 
            prior : list[torch.Tensor | list[float]] | None=None, 
            **kwargs
        ):
        """TODO.

        Args:
            sparse_masks: Long-Tensors with parent indices for each element in layers n-1.
            masks: DEPRECATED! Dense child-parent "log-adjacency" matrices.
            kwargs: passed to `mini_trainer.classifier.Classifier`.
        """
        super().__init__(**kwargs)
        if not self.normalized:
            raise NotImplementedError("Unnormalized hierarchical models is not implemented yet!")

        if prior is not None:
            self._metadata["prior"] = [p.tolist() if isinstance(p, torch.Tensor) else p for p in prior]
        
        if self._metadata.get("prior", None) is not None:
            self.linear.bias.data[:] = torch.tensor(
                data=self._metadata["prior"][0], 
                device=self.linear.weight.device, 
                dtype=self.linear.weight.dtype
            )

        # Store masks
        self._num_masks = 0
        if sparse_masks is not None:
            for i, m in enumerate(sparse_masks):
                self.register_buffer(f"mask_{i}", m.long(), persistent=True)
                self.register_buffer(f"_mask_{i}", torch.empty(0), persistent=False)
                self.register_buffer(f"_filter_{i}", torch.empty(0), persistent=False)
            self._num_masks = len(sparse_masks)
            self.register_buffer(f"_filter_{self.num_masks}", torch.empty(0), persistent=False)
        _ = self.masks # Call the masks attribute to "warm up" the mask-related buffer cache

    @classmethod
    def load(
        cls, 
        architecture_class, 
        architecture_output_name, 
        architecture, 
        state, 
        device, 
        dtype, 
        cls2idx : dict[str, dict[str, int]] | None=None, 
        train_labels : list[list[int]] | None=None, 
        **kwargs
    ):
        if state is not None:
            prefix = f"{architecture_output_name}.mask_"
            # Extract (index, tensor) tuples, sort by index, and retrieve tensors
            masks = [(int(k.split("_")[-1]), v) for k, v in state.items() if k.startswith(prefix)]
            if masks:
                kwargs["sparse_masks"] = [v for _, v in sorted(masks)]
        if cls2idx is not None:
            kwargs["cls2idx"] = cls2idx
            if train_labels is not None:
                kwargs["prior"] = prior_from_labels(train_labels, cls2idx=cls2idx)
        return super().load(architecture_class, architecture_output_name, architecture, state, device, dtype, **kwargs)

    @property
    def num_masks(self):
        stored = getattr(self, "_num_masks", None)
        if not isinstance(stored, int):
            i = 0
            while hasattr(self, f"mask_{i}"):
                i += 1
            self._num_masks = stored = i
        return stored

    @property
    def masks(self):
        if self._dirty_cache["_masks"]:
            masks = []
            filter = self.active_indices
            filters = [filter]
            for i in range(self.num_masks):
                mask = getattr(self, f"mask_{i}")
                setattr(self, f"_filter_{i}", None if filter is None else filter.view_as(filter))
                if filter is not None:
                    mask = mask[filter]
                    filter, mask = mask.unique(sorted=False, return_inverse=True)
                setattr(self, f"_mask_{i}", mask.view_as(mask))
                masks.append(mask)
                filters.append(filter)
            setattr(self, f"_filter_{self.num_masks}", None if filter is None else filter.view_as(filter))
            self._dirty_cache["_masks"] = False
        masks : list[torch.Tensor] = []
        for i in range(self.num_masks):
            masks.append(self.mask(i))
        return masks
    
    def mask(self, idx : int) -> torch.Tensor:
        if self._dirty_cache["_masks"]:
            _ = self.masks
        return getattr(self, f"_mask_{idx}")

    def hierarchy(self, log_probs : torch.Tensor):
        ys = [log_probs]
        # Propagate the probabilities up the hierarchy using the masks
        for mask in self.masks:
            ys.append(batched_scatter_logsumexp(ys[-1], mask))
        return ys

    def forward(self, x):
        return self.hierarchy(super().forward(x))

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
        return HierarchicalPrediction(self(x), topk=topk, **self._preprocess_metadata(**kwargs))


class ConditionalClassifier(HierarchicalClassifier): # noqa: D101 TODO
    def __init__(self, normalized : bool=True, **kwargs): # noqa: D107
        super().__init__(normalized=normalized, **kwargs)
        # Conditional layers
        layers : list[nn.Linear] = [self.linear]
        orig_indices = self.active_indices
        self.active_indices = None
        for i, m in enumerate(self.masks):
            out = int(m.max().item() + 1)
            layer = nn.Linear(self.preclassification_size, out, bias=True)
            layer = self._normalize_layer(layer) if normalized else layer
            if self._metadata.get("prior", None) is not None:
                layer.bias.data[:] = torch.tensor(
                    data=self._metadata["prior"][i + 1], 
                    device=layer.weight.device, 
                    dtype=layer.weight.dtype
                )
            layers.append(layer)
        self.active_indices = orig_indices
        self.layers = nn.ModuleList(layers)
        for i in range(len(self.layers)):
            self.register_buffer(f"_linear_weight_{i}", torch.empty(0), persistent=False)
            self.register_buffer(f"_linear_bias_{i}", torch.empty(0), persistent=False)

    def _weight_bias(self, i : int) -> tuple[torch.Tensor, torch.Tensor]:
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
            setattr(self, f"_linear_weight_{i}", weight.view_as(weight))
            setattr(self, f"_linear_bias_{i}", bias.view_as(bias))
            self._dirty_cache[f"_weight_bias_{i}"] = False
        return getattr(self, f"_linear_weight_{i}"), getattr(self, f"_linear_bias_{i}")

    def marginals(self, x : torch.Tensor) -> list[torch.Tensor]:
        M = []
        for i in range(len(self.layers)):
            w, b = self._weight_bias(i)
            L = cosine_to_zscore(
                F.linear(x, w), 
                self.preclassification_size
            ) + b
            M.append(L)
        return M

    def forward(self, x : torch.Tensor):
        embeddings = self.preclassification(x)
        if EmbeddingContext.active():
            EmbeddingContext.set(embeddings)
        M = self.marginals(embeddings)
        N = len(M)
        C : list[torch.Tensor] = [torch.empty(0) for _ in range(N)]
        C[-1] = M[-1] # Top-level classes are not conditioned
        for i in reversed(range(N - 1)):
            parent_logprop = C[i + 1]
            sibling_norm = batched_scatter_logsumexp(M[i], self.mask(i))
            # Top-down condition : P_cond(x) = P(x) * P_cond(parent(x)) / P(siblings(x))
            # ==> log(P_cond(x)) = log(P(x)) + log(P_cond(parent(x))) - log(P(siblings(x)))
            C[i] += (parent_logprop - sibling_norm).gather(1, self.mask(i).expand_as(M[i]))
        return C


class IndependentClassifier(ConditionalClassifier): # noqa: D101 TODO
    def forward(self, x):
        embeddings = self.preclassification(x)
        if EmbeddingContext.active():
            EmbeddingContext.set(embeddings)
        return self.marginals(embeddings)


class AutoregressiveClassifier(IndependentClassifier): # noqa: D101 TODO
    def __init__(self, *args, **kwargs):  # noqa: D107
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
        M = []
        for i, x in list(enumerate(sequence[::-1])):
            w, b = self._weight_bias(i)
            L = cosine_to_zscore(
                F.linear(F.normalize(x, 2, 1), w), 
                self.preclassification_size
            ) + b
            M.append(L)
        return M

    def forward(self, x : torch.Tensor, y : torch.Tensor | list[int] | None=None):
        if y is None:
            # Check if a label is passed around parent module via context manager
            y = SupervisionContext.get()

        # Image embedding context
        context = self.preclassification(x)
        if EmbeddingContext.active():
            EmbeddingContext.set(context)

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
                decision[i + 1] = cosine_to_zscore(
                    F.normalize(sequence[i], 2, 1) @ self.embedding(di).T,
                    self.preclassification_size
                ).softmax(dim=1) @ self.embedding(di)
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

        return self._classify(sequence[1:])


# Differs from the one above in that we don't carry explicit independent embeddings for each layer
class AutoregressiveClassifierV2(HierarchicalClassifier): # noqa: D101 TODO
    def __init__(self, *args, **kwargs):  # noqa: D107
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
        M = []
        w, b = self._weight_bias()
        for i, x in list(enumerate(sequence[::-1])):
            L = cosine_to_zscore(
                F.linear(F.normalize(x, 2, 1), w), 
                self.preclassification_size
            ) + b
            M.append(self.hierarchy(L)[i])
        return M

    def forward(self, x : torch.Tensor, y : torch.Tensor | list[int] | None=None):
        if y is None:
            # Check if a label is passed around parent module via context manager
            y = SupervisionContext.get()

        # Image embedding context
        context = self.preclassification(x)
        if EmbeddingContext.active():
            EmbeddingContext.set(context)

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
                logits = cosine_to_zscore(
                    F.normalize(sequence[i], 2, 1) @ self.embeddings.T,
                    self.preclassification_size
                )
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
                decision[i + 1] = logits.softmax(dim=1) @ self.embeddings
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
                    dist = torch.zeros((batch_size, len(self.mask(0))), device=device, dtype=dtype, requires_grad=False)
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

        return self._classify(sequence[1:])


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
            data_str = " / ".join([f"{label}-({conf:.1%})" for conf, label in zip(self.confidence, self.label)])
        else:
            data_str = " / ".join([f"I[{idx}]-({conf:.1%})" for conf, idx in zip(self.confidence, self.index)])
        return f'| {data_str} |'


class HierarchicalPrediction(BasePrediction[HierarchicalPredictionItem, list[torch.Tensor]]):
    """Hierarchical PyTorch Prediction class."""

    ITEM_CLASS = HierarchicalPredictionItem

    @property
    @lru_cache(1)
    def idx2cls(self) -> dict[str, dict[int, str]] | None:
        if self.cls2idx:
            return {level : {v: k for k, v in mapping.items()} for level, mapping in self.cls2idx.items()}
        return None

    def _process(self, raw_prediction):
        logits, indices = [], []
        for rp in raw_prediction:
            dim = rp.shape[-1]
            if self.topk > dim:
                raise RuntimeError(
                    f'{self.topk=} must be less than the number of classes in the smallest layer in the hierarchy.'
                )
            lgs, idx = torch.topk(rp, self.topk)
            logits.append(lgs)
            indices.append(idx)
        return (
            torch.stack([torch.stack(v, dim=1) for v in zip(*logits)]), 
            torch.stack([torch.stack(v, dim=1) for v in zip(*indices)])
        )

    def _translate(self):
        idx2cls = self.idx2cls
        if idx2cls is not None:
            def fmt_idx(level, i): 
                return idx2cls[str(level)][i.item()]
        else:
            def fmt_idx(level, i):
                return f"{i.item()}[{level}]"
        return [[[fmt_idx(level, i) for level, i in enumerate(idxs)] for idxs in e] for e in self.indices]

    def _extract_confidence(self, raw_prediction) -> list[torch.Tensor]:
        confidences = []
        for rp, idx in zip(raw_prediction, torch.permute(self.indices, (2, 0, 1))):
            if not (
                torch.all(rp >= 0) and
                torch.isclose(
                    rp.sum(), 
                    torch.tensor(1.0, dtype=rp.dtype), 
                    atol=1e-3 if rp.dtype == torch.float16 else 1e-5
                )
            ):
                rp = rp.softmax(dim=-1)
            confidences.append(rp.gather(-1, idx))
        return torch.stack([torch.stack(v, dim=1) for v in zip(*confidences)])
