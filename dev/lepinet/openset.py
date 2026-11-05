"""Open-set scores: per image and rank, summaries of the head's native (pre-softmax) output.

    python dev/lepinet/openset.py --weights <last.pt> --input <gbifxdl.parquet> --output <dir> --name openset

Takes ``mt_hpredict``'s arguments. Writes ``<dir>/<name>/summary.parquet`` with ``path``, ``label_<r>`` and, per
rank r, ``z1_<r>`` and ``z2_<r>`` (top two values), ``lse_<r>`` (logsumexp), ``entropy_<r>`` (softmax entropy) and
``argmax_<r>`` (class index), plus ``classes.json`` (the model's ``cls2idx``). That is what the max-logit, energy,
MSP, entropy and margin rules need, without the full log-probabilities (about 15 GB per rank on set 0).
"""

import json
import os
from collections import defaultdict

import numpy as np
import torch

from mini_trainer.hierarchical.predict import cli
from mini_trainer.logging import RawResultCollector
from mini_trainer.modeling import classification_module
from mini_trainer.predict import main as predict


class OpenSetSummaryCollector(RawResultCollector):
    """Keep five per-rank summaries of each image's raw output instead of the output itself."""

    def __init__(self, model: torch.nn.Module | None = None, **kwargs):  # noqa: D107
        super().__init__()
        self.cls2idx = classification_module(model).metadata.get("cls2idx") if model is not None else None
        self.columns = defaultdict(list)

    def collect(self, paths=None, predictions=None, labels=None, **kwargs):  # noqa: D102
        outputs = list(predictions) if isinstance(predictions, (list, tuple)) else [predictions]
        for rank, output in enumerate(outputs):
            z = output.float()
            top = z.topk(2, dim=-1)
            lse = torch.logsumexp(z, dim=-1)
            log_p = z - lse[:, None]
            summary = {"z1": top.values[:, 0], "z2": top.values[:, 1], "lse": lse, "entropy": -(log_p.exp() * log_p).sum(-1)}
            for key, value in summary.items():
                self.columns[f"{key}_{rank}"].append(value.cpu().numpy())
            self.columns[f"argmax_{rank}"].append(top.indices[:, 0].cpu().numpy())
        count = len(outputs[0])
        self.columns["path"].append(np.array(paths if paths is not None else [None] * count, dtype=object))
        if labels is not None and len(labels):
            per_rank = zip(*labels) if isinstance(labels[0], (list, tuple)) else [labels]
            for rank, values in enumerate(per_rank):
                self.columns[f"label_{rank}"].append(np.array([str(v) for v in values], dtype=object))

    def save(self, dst: str, *args, **kwargs):  # noqa: D102
        import pyarrow as pa
        import pyarrow.parquet as pq

        os.makedirs(dst, exist_ok=True)
        if os.path.exists(os.path.join(dst, "summary.parquet")):
            raise FileExistsError(f"Prediction outputs already exist in {dst}")
        order = ["path", *sorted(k for k in self.columns if k.startswith("label_"))]
        order += sorted(k for k in self.columns if k not in order)
        table = pa.table({k: np.concatenate(self.columns[k]) for k in order})
        pq.write_table(table, os.path.join(dst, "summary.parquet"))
        with open(os.path.join(dst, "classes.json"), "w") as handle:
            json.dump(self.cls2idx, handle)


if __name__ == "__main__":
    kwargs = cli(description="Open-set summaries of a hierarchical model's raw outputs")
    kwargs["collector_cls"] = OpenSetSummaryCollector
    predict(**kwargs)
