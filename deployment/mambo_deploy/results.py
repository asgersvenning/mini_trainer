"""Runtime-neutral result containers and masked hierarchy postprocessing."""

import json
from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class PredictionItem:
    label: tuple[str, ...]
    confidence: tuple[float, ...]
    index: tuple[int, ...]

    def to_dict(self):
        return asdict(self)


class HierarchyPlan:
    """Cache vocabulary order and parent groups; no runtime dependency for NumPy."""

    def __init__(self, selected, classes):
        indices = np.asarray(selected, dtype=np.int64)
        self.full = np.array_equal(indices, np.arange(len(classes["labels"][0])))
        self.indices = [indices]
        self.groups = []
        for parents in classes["parents"]:
            indices, inverse = np.unique(np.asarray(parents, dtype=np.int64)[indices], return_inverse=True)
            order = np.argsort(inverse, kind="stable")
            starts = np.r_[0, np.flatnonzero(np.diff(inverse[order])) + 1]
            self.groups.append((inverse, order, starts))
            self.indices.append(indices)
        self.labels = [[classes["labels"][rank][int(i)] for i in indices] for rank, indices in enumerate(self.indices)]
        self._devices = {}

    def numpy(self, leaf):
        values = np.asarray(leaf if self.full else leaf[:, self.indices[0]], dtype=np.float32)
        logits = [values]
        for inverse, order, starts in self.groups:
            ordered = values[:, order]
            maxima = np.maximum.reduceat(ordered, starts, axis=1)
            shifts = np.where(np.isfinite(maxima), maxima, 0)
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                shifted = np.exp(ordered - shifts[:, inverse[order]])
                values = np.log(np.add.reduceat(shifted, starts, axis=1)) + shifts
            logits.append(values)
        return logits, self.labels, self.indices

    def torch(self, leaf, native=None):
        import torch

        from mini_trainer.hierarchical.utils import batched_scatter_logsumexp

        if self.full and native is not None:
            values = native
        else:
            key = str(leaf.device)
            if key not in self._devices:
                self._devices[key] = (
                    torch.as_tensor(self.indices[0], device=leaf.device),
                    [torch.as_tensor(group[0], device=leaf.device) for group in self.groups],
                )
            selected, parents = self._devices[key]
            values = [leaf.index_select(1, selected)]
            for rank, index in enumerate(parents, start=1):
                values.append(batched_scatter_logsumexp(values[-1], index, dim_size=len(self.indices[rank])))
        return [value.float().cpu().numpy() for value in values], self.labels, self.indices


def hierarchy(leaf, selected, classes):
    """Mask leaves before batched stable parent reduction, preserving class order."""
    return HierarchyPlan(selected, classes).numpy(leaf)


class Prediction:
    def __init__(self, raw, labels, global_indices, topk=1, **metadata):
        if not isinstance(topk, int) or topk < 1 or topk > min(map(len, labels)):
            raise ValueError("topk must be positive and no larger than the smallest retained rank")
        self.topk, self.metadata, self.raw_logits = topk, metadata, raw
        self.cls2idx = {str(rank): {label: i for i, label in enumerate(names)} for rank, names in enumerate(labels)}
        indices = [
            np.argmax(values, axis=1)[:, None]
            if topk == 1 and not np.isnan(values).any()
            else np.argsort(-values, axis=1, kind="stable")[:, :topk]
            for values in raw
        ]
        self.indices = np.stack(indices, axis=-1)
        self.global_indices = np.stack([mapping[idx] for mapping, idx in zip(global_indices, indices)], axis=-1)
        self.logits = np.stack([np.take_along_axis(values, idx, axis=1) for values, idx in zip(raw, indices)], axis=-1)
        probabilities = []
        for values, idx in zip(raw, indices):
            exp = np.exp(values - values.max(axis=1, keepdims=True))
            probabilities.append(np.take_along_axis(exp / exp.sum(axis=1, keepdims=True), idx, axis=1))
        self.confidence = np.stack(probabilities, axis=-1)
        self.labels = [[[labels[r][int(self.indices[b, k, r])] for r in range(3)] for k in range(topk)] for b in range(len(raw[0]))]
        nested = [
            [PredictionItem(tuple(lab), tuple(map(float, conf)), tuple(map(int, idx))) for lab, conf, idx in zip(labs, confs, idxs)]
            for labs, confs, idxs in zip(self.labels, self.confidence, self.indices)
        ]
        self.items = [row[0] for row in nested] if topk == 1 else nested

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]

    def __iter__(self):
        return iter(self.items)

    def to_dict(self):
        return [item.to_dict() for item in self.items] if self.topk == 1 else [[item.to_dict() for item in row] for row in self.items]

    def save(self, path):
        with open(path, "w") as stream:
            json.dump(
                {"results": self.to_dict(), "metadata": self.metadata, "config": {"topk": self.topk, "cls2idx": self.cls2idx}}, stream
            )
