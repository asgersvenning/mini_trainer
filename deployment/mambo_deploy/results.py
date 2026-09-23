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


def hierarchy(leaf, selected, classes):
    """Mask leaves first; preserve original rank ordering and recompute every parent."""
    indices = np.asarray(selected, dtype=np.int64)
    values = np.asarray(leaf[:, indices], dtype=np.float32)
    logits, global_indices = [values], [indices]
    for parents in classes["parents"]:
        selected_parents = np.asarray(parents, dtype=np.int64)[indices]
        indices, inverse = np.unique(selected_parents, return_inverse=True)
        grouped = np.full((len(values), len(indices)), -np.inf, dtype=np.float32)
        for row in range(len(values)):
            np.logaddexp.at(grouped[row], inverse, values[row])
        values = grouped
        logits.append(values)
        global_indices.append(indices)
    labels = [[classes["labels"][rank][int(i)] for i in indices] for rank, indices in enumerate(global_indices)]
    return logits, labels, global_indices


class Prediction:
    def __init__(self, raw, labels, global_indices, topk=1, **metadata):
        if not isinstance(topk, int) or topk < 1 or topk > min(map(len, labels)):
            raise ValueError("topk must be positive and no larger than the smallest retained rank")
        self.topk, self.metadata, self.raw_logits = topk, metadata, raw
        self.cls2idx = {str(rank): {label: i for i, label in enumerate(names)} for rank, names in enumerate(labels)}
        indices = [np.argsort(-values, axis=1, kind="stable")[:, :topk] for values in raw]
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
