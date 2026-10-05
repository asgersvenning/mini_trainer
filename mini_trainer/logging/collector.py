import json
import os
import shutil
import tempfile
from collections import defaultdict
from itertools import repeat
from types import GeneratorType
from typing import cast

import numpy as np
import torch

from mini_trainer.integrations import id_to_name
from mini_trainer.modeling import EmbeddingContext, Prediction, classification_module
from mini_trainer.training import named_confusion_matrix
from mini_trainer.utils import write_csv_from_dict
from mini_trainer.visualization import plot_heatmap


class _ResultsCollector:
    """This is an abstract base class, it is likely easier to subclass `BaseResultCollector` instead.

    If you are using the `mini_trainer` train and prediction scripts/APIs,
      it is very unlikely that this is the correct class to subclass.
    However, if you are building entirely new train and/or predictions scripts/APIs, it may be an option.
    """

    def collect(self, *args, **kwargs):
        raise NotImplementedError("Result collectors must have a `collect` class method.")

    def evaluate(self):
        raise NotImplementedError("Result collector must have a `evaluate` class method.")

    @property
    def data(self):
        raise NotImplementedError("Result collector must have `data` class propery suitable for JSON serialization.")


_BaseTypes = bool | str | float | int | torch.Tensor | np.ndarray | np.str_


class RawResultCollector(_ResultsCollector):
    """Agnostic collector with minimal postprocessing."""

    _attributes = ("predictions", "labels", "paths")

    def __init__(  # noqa: D107
        self, strict: bool = True, *args, **kwargs
    ):
        self.strict = strict
        self.predictions, self.labels, self.paths = [], [], []

    def collect(
        self,
        paths: list[str] | None = None,
        predictions: torch.Tensor | list[torch.Tensor] | None = None,
        labels: list[int] | list[list[int]] | None = None,
        **kwargs,
    ):
        contrib = locals()
        for attr in self._attributes:
            try:
                values = contrib.get(attr, None)
                if isinstance(values, torch.Tensor):
                    values = values.detach().cpu()
                if values is not None and len(values) > 0:
                    if not isinstance(values, (torch.Tensor, np.ndarray)):
                        if isinstance(values[0], torch.Tensor):
                            values = [v.detach().cpu() for v in values]
                        if isinstance(values[0], torch.Tensor) and values[0].ndim <= 1:
                            values = torch.stack(values)
                        elif isinstance(values[0], (np.ndarray, np.str_)) and values[0].ndim <= 1:
                            values = np.stack(values)
                    if len(getattr(self, attr)) > 0:
                        assert isinstance(getattr(self, attr)[0], type(values))
                    getattr(self, attr).append(values)
            except Exception as e:
                raise RuntimeError(f"Error while collecting {attr}.") from e

    def _stack_and_normalize(self, data: list[_BaseTypes | list[_BaseTypes]] | torch.Tensor | np.ndarray | list[str]):
        if len(data) < 1:
            return data
        if isinstance(data[0], (list, tuple)):
            return [self._stack_and_normalize(col) for col in zip(*data)]
        assert isinstance(data[0], _BaseTypes)
        if data and isinstance(data[0], torch.Tensor):
            data = torch.cat(cast(list[torch.Tensor], data))
        elif data and isinstance(data[0], np.ndarray):
            data = np.concat(cast(list[np.ndarray], data))
        if isinstance(data, np.ndarray):
            if data.dtype.type is np.str_:
                data = cast(list[str], data.tolist())  # ty:ignore[no-matching-overload]
            else:
                data = torch.from_numpy(data)
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], (float, int, bool)):
            data = torch.tensor(data)
        return data

    def _datalength(self, data: torch.Tensor | list[str] | list[torch.Tensor] | list[list[str]]):
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], torch.Tensor):
            lengths = list(map(len, data))
            assert len(set(lengths)) == 1
            return lengths[0]
        return len(data)

    @property
    def data(self):
        data = {}
        for attr in self._attributes:
            values = getattr(self, attr)
            data[attr] = self._stack_and_normalize(values)
        data_length = {k: self._datalength(v) for k, v in data.items()}
        non_empty = [k for k, length in data_length.items() if length > 0]
        if self.strict and len(non_empty) == 0:
            raise RuntimeError(f"Attempt to access empty data: {data_length}")
        mdl = max(data_length.values())
        consistent = [data_length[k] == mdl for k in non_empty]
        if self.strict and not all(consistent):
            raise RuntimeError(f"Stored data is heterogeneous: {data_length}")
        return data

    def save(self, dst: str, *args, **kwargs):
        if os.path.isdir(dst):
            dst = os.path.join(dst, "predictions.pt")
        torch.save(self.data, dst)


class ParquetResultCollector(RawResultCollector):
    """Stream every rank's float16 log-probabilities, and optionally embeddings, to Parquet shards.

    ``save(dst)`` writes ``rank-<r>/part-<k>.parquet`` with columns ``row, c0..cK`` (log-softmax of
    each rank's native output), ``embeddings/part-<k>.parquet`` with ``row, e0..eD`` (the tensor the
    head publishes through ``EmbeddingContext``), ``index.parquet`` with ``row, path`` and any
    ``label_<r>`` columns, and ``classes.json`` with the model's class indices. Shards are buffered
    in a temporary directory so memory stays bounded by ``shard_size`` rows.
    """

    def __init__(self, model: torch.nn.Module | None = None, embeddings: bool = True, shard_size: int = 8192, **kwargs):
        try:
            import pyarrow  # noqa: F401
        except ImportError as error:
            raise ImportError("ParquetResultCollector requires pyarrow. Install with `pip install mt-trainer[recommended]`.") from error
        super().__init__()
        self.embeddings, self.shard_size = embeddings, shard_size
        self.cls2idx = classification_module(model).metadata.get("cls2idx") if model is not None else None
        self._directory = tempfile.mkdtemp(prefix="mt-predictions-")
        self._rows, self._shards, self._buffer, self._index = 0, 0, defaultdict(list), defaultdict(list)

    def collect(self, paths=None, predictions=None, labels=None, **kwargs):
        outputs = list(predictions) if isinstance(predictions, (list, tuple)) else [predictions]
        for rank, output in enumerate(outputs):
            self._buffer[f"rank-{rank}"].append(torch.log_softmax(output.float(), -1).to(torch.float16).cpu().numpy())
        embeddings = EmbeddingContext.get() if self.embeddings else None
        if embeddings is not None:
            self._buffer["embeddings"].append(embeddings.detach().to(torch.float16).cpu().numpy())
        count = len(outputs[0])
        self._index["row"].extend(range(self._rows, self._rows + count))
        self._index["path"].extend(paths if paths is not None else [None] * count)
        if labels is not None and len(labels):
            per_rank = zip(*labels) if isinstance(labels[0], (list, tuple)) else [labels]
            for rank, values in enumerate(per_rank):
                self._index[f"label_{rank}"].extend(values)
        self._rows += count
        if sum(len(part) for part in self._buffer["rank-0"]) >= self.shard_size:
            self._flush()

    def _flush(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        if not self._buffer["rank-0"]:
            return
        for name, parts in self._buffer.items():
            values = np.concatenate(parts)
            first = self._rows - len(values)
            prefix = "e" if name == "embeddings" else "c"
            columns = {"row": pa.array(np.arange(first, self._rows))}
            columns |= {f"{prefix}{j}": pa.array(values[:, j]) for j in range(values.shape[1])}
            os.makedirs(os.path.join(self._directory, name), exist_ok=True)
            pq.write_table(pa.table(columns), os.path.join(self._directory, name, f"part-{self._shards:05d}.parquet"))
        self._buffer.clear()
        self._shards += 1

    def save(self, dst: str, *args, **kwargs):
        import pyarrow as pa
        import pyarrow.parquet as pq

        self._flush()
        os.makedirs(dst, exist_ok=True)
        # Prediction writes its resolved config here first; refuse only to overwrite our outputs.
        if any(os.path.exists(os.path.join(dst, name)) for name in ("index.parquet", "classes.json", "rank-0", "embeddings")):
            raise FileExistsError(f"Prediction outputs already exist in {dst}")
        pq.write_table(pa.table(dict(self._index)), os.path.join(dst, "index.parquet"))
        with open(os.path.join(dst, "classes.json"), "w") as handle:
            json.dump(self.cls2idx, handle)
        for name in sorted(os.listdir(self._directory)):
            shutil.move(os.path.join(self._directory, name), os.path.join(dst, name))
        shutil.rmtree(self._directory, ignore_errors=True)


class BaseResultCollector(_ResultsCollector):
    def __init__(
        self,
        model: torch.nn.Module | None = None,
        idx2cls: dict | None = None,
        cls2idx: dict | None = None,
        verbose: bool = False,
        scientific_names: bool = True,
        additional_attributes: list[str] | None = None,
        *args,
        **kwargs,
    ):
        self.scientific_names = scientific_names
        self._sn_cache = defaultdict(str)

        if model is not None:
            model_metadata = classification_module(model).metadata
            cls2idx = model_metadata.get("cls2idx", None)
            if cls2idx is not None:
                idx2cls = None

        if cls2idx is not None and self.scientific_names:
            cls2idx = self._cls2idx_to_scientific(cls2idx)
            idx2cls = None

        if idx2cls is None and cls2idx is None:
            raise ValueError("Either `idx2cls` or `cls2idx` must not be `None`.")

        if cls2idx is None:
            cls2idx = self._invert_mapping(idx2cls)
        if idx2cls is None:
            idx2cls = self._invert_mapping(cls2idx)

        self.paths = []
        self.preds = []
        self.confs = []
        self.labels = []
        self.cls2idx, self.idx2cls = cls2idx, idx2cls
        self.verbose = verbose
        self._extra_attr = set(additional_attributes or [])
        for attr in self._extra_attr:
            setattr(self, attr, [])

    # --- Hook Methods for Polymorphic Initialization ---
    def _cls2idx_to_scientific(self, cls2idx: dict) -> dict:
        return {self._get_scientific_name(k): v for k, v in cls2idx.items()}

    def _get_scientific_name(self, name: str) -> str:
        if name in self._sn_cache:
            return self._sn_cache[name]
        return self._sn_cache.setdefault(name, id_to_name(name))

    def _invert_mapping(self, mapping: dict) -> dict:
        return {v: k for k, v in mapping.items()}

    # --- Collection Methods ---
    def collect(self, paths: list[str], predictions: torch.Tensor | Prediction, labels: list[int | str] | None = None, **kwargs):
        self._collect_base_attributes(paths, predictions, labels)
        self._collect_extra_attributes(**kwargs)

    def _collect_base_attributes(
        self, paths: list[str], predictions: torch.Tensor | Prediction, labels: list[int | str] | list[list[int | str]] | None = None
    ):
        self.paths.extend(paths)

        if not isinstance(predictions, Prediction):
            predictions = Prediction(predictions, topk=1, cls2idx=self.cls2idx)
        self.preds.extend([self.idx2cls[p.index] for p in predictions])
        self.confs.extend([p.confidence for p in predictions])

        if labels is not None:
            processed_labels = [str(e) if isinstance(e, (str, int)) else str(e[0]) for e in labels]
            if self.scientific_names:
                processed_labels = [self._get_scientific_name(e) for e in processed_labels]
            self.labels.extend(processed_labels)

    def _collect_extra_attributes(self, **kwargs: list | tuple | GeneratorType | np.ndarray | torch.Tensor):
        if len(self._extra_attr) == 0:
            return
        if not all([attr in kwargs for attr in self._extra_attr]):
            raise ValueError(
                "To ensure proper ordering and avoid data loss it is required "
                f"to always pass all extra attributes ([{', '.join(self._extra_attr)}])"
            )
        for key, value in kwargs.items():
            if value is None:
                continue
            elif isinstance(value, list):
                pass
            elif isinstance(value, (torch.Tensor, np.ndarray)):
                value = value.tolist()  # ty:ignore[no-matching-overload]
                if not isinstance(value, list):
                    raise ValueError(
                        f"Value passed for {key} is likely a zero-dimensional (scalar) "
                        f"array/tensor containing a {type(value)}.\nIf you want to pass "
                        "a single value, it should still be contained in a 1-dimensional "
                        "array/tensor:\n\tIncorrect: `torch.tensor(1)`/`np.array(1)`\n"
                        "\tCorrect: `torch.tensor([1])`/`np.array([1])`"
                    )
            elif isinstance(value, (tuple, GeneratorType)):
                value = list(value)
            else:
                raise TypeError(f"Unexpected value type `{type(value)}` supplied for {key}.")
            getattr(self, key).extend(value)

    # --- Evaluation Methods ---
    def eval_label_fn(self, data: dict, outdir: str | None, save: bool, prefix: str = "", plot_conf_mat: bool = False, **kwargs):
        if kwargs:
            raise RuntimeError(
                f"Unknown arguments ([{', '.join(kwargs)}]) passed."
                "Perhaps you forgot to implement the intended `eval_label_fn` in your subclass."
            )
        if save and not isinstance(outdir, str):
            raise RuntimeError("Attempted to save evaluated results against labels without specifying an output directory.")
        dst = os.path.join(outdir, f"{prefix}confusion_matrix.png") if plot_conf_mat and save else None
        return self._evaluate_labels(data, self.cls2idx, dst)

    def _evaluate_labels(self, data: dict, cls2idx: dict, dst: str | None):
        """Evaluate one vocabulary and optionally save its count matrix."""
        results = named_confusion_matrix(data, cls2idx, verbose=self.verbose)
        if dst is not None:
            from PIL.Image import fromarray

            classes = sorted(cls2idx, key=cls2idx.get)
            conf_mat = results["conf_mat"]
            counts = np.array([[conf_mat[label][pred] for pred in classes] for label in classes], dtype=np.float64)
            fromarray(plot_heatmap(counts, "magma", percent=False)).save(dst)
        return results

    def evaluate(self, outdir: str | None = None, prefix: str = "", **kwargs):
        do_save = isinstance(outdir, str)
        if do_save and not os.path.isdir(outdir):
            raise OSError(f"Specified output directory (`{outdir}`) does not exist.")
        if self.labels:
            results = self.eval_label_fn(data=self.data, outdir=outdir, save=do_save, prefix=prefix, **kwargs)
            if do_save:
                with open(os.path.join(outdir, f"{prefix}eval_results.json"), "w") as f:
                    json.dump(results, f)
            return results

    @property
    def data(self):
        return {
            "paths": self.paths,
            "preds": self.preds,
            "confs": self.confs,
            "labels": self.labels,
            **{attr: getattr(self, attr) for attr in self._extra_attr},
        }

    # --- Hooks for Generic Flat/Hierarchical Saving ---
    def _is_known_label(self, label: str, level: int = 0) -> bool:
        return label in self.cls2idx

    def _get_evaluation_rows(self):
        labels = self.labels or repeat("-1")
        for i, (path, pred, lab, conf) in enumerate(zip(self.paths, self.preds, labels, self.confs)):
            yield i, path, 0, lab, pred, conf

    def save(self, dst: str, threshold: float = 0.0):
        if os.path.isdir(dst):
            dst = os.path.join(dst, "mini_metric.csv")
        SCHEMA = {
            "instance_id": int,
            "filename": str,
            "level": int,
            "label": str,
            "prediction": str,
            "confidence": float,
            "threshold": float,
            "known_label": int,
            "prediction_made": int,
            "correct": int,
        }
        data = {k: list() for k in SCHEMA}

        for i, path, level, label, pred, conf in self._get_evaluation_rows():
            do_predict = int(conf >= threshold)
            row = {
                "instance_id": i,
                "filename": path,
                "level": level,
                "label": label,
                "prediction": pred,
                "confidence": conf,
                "threshold": float(threshold),
                "known_label": int(self._is_known_label(label, level)),
                "prediction_made": do_predict,
                "correct": do_predict if do_predict == 0 else 1 if pred == label else -1,
            }
            for k, v in row.items():
                if not isinstance(v, SCHEMA[k]):
                    raise RuntimeError(f"Invalid data type in {k}, found {v}, but expected a {SCHEMA[k]}")
                data[k].append(v)
        write_csv_from_dict(data, dst)
