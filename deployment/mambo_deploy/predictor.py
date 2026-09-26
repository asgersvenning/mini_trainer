"""Two explicit backends sharing one image, vocabulary and result contract."""

import hashlib
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, nullcontext
from functools import cached_property
from itertools import islice
from pathlib import Path

import numpy as np

from .augmentation import infer_augmented, infer_prepared, prepared_views, resolve_tta
from .bundle import Bundle
from .download import default_bundle
from .onnx_session import create_session
from .preprocessing import RECIPE, TorchDecode, TorchPreprocess, _rgb, image_items, prepare_batch
from .result_worker import ResultWorker
from .results import HierarchyPlan, Prediction
from .streaming import prepared_stream
from .transfers import device_batches, download_tensors, pinned_factory


class Predictor:
    def __init__(
        self,
        bundle=None,
        *,
        backend="onnx",
        device="cpu",
        model=None,
        class_list=None,
        class_mask=None,
        weights=None,
        batch_size=8,
        threads=2,
        precision="auto",
        tta="none",
        preprocess_workers=None,
    ):
        if backend not in ("torch", "onnx"):
            raise ValueError("backend must be 'torch' or 'onnx'")
        if device != "cpu" and not re.fullmatch(r"cuda(?::\d+)?", str(device)):
            raise ValueError("device must be cpu, cuda or cuda:N")
        if not isinstance(batch_size, int) or batch_size < 1 or not isinstance(threads, int) or threads < 1:
            raise ValueError("batch_size and threads must be positive integers")
        self.tta = resolve_tta(tta)
        self.preprocess_workers = threads if preprocess_workers is None else preprocess_workers
        if not isinstance(self.preprocess_workers, int) or self.preprocess_workers < 1:
            raise ValueError("preprocess_workers must be a positive integer")
        if precision not in ("auto", "fp32", "fp16", "bf16", "tf32"):
            raise ValueError("precision must be auto, fp32, fp16, bf16 or tf32")
        self.precision = precision
        self.effective_precision = precision
        if precision == "auto":
            self.effective_precision = ("fp16" if backend == "torch" else "tf32") if str(device).startswith("cuda") else "fp32"
        if self.effective_precision != "fp32" and device == "cpu":
            raise ValueError("CPU inference requires auto or fp32 precision")
        if backend == "onnx" and self.effective_precision not in ("fp32", "tf32"):
            raise ValueError("Standard ONNX supports fp32 or CUDA tf32; fp16/bf16 autocast requires the torch backend")
        if backend == "torch" and self.effective_precision == "tf32":
            raise ValueError("tf32 is an ONNX CUDA option; use fp16, bf16 or fp32 for torch")
        if class_list is not None and class_mask is not None:
            raise ValueError("class_list and class_mask are mutually exclusive")
        if weights is not None and model is not None:
            raise ValueError("model and weights are mutually exclusive")
        if weights is not None and backend != "torch":
            raise ValueError("weights override is only supported by the PyTorch backend")
        bundle = bundle or os.environ.get("MAMBO_BUNDLE")
        automatic = not bundle
        self.bundle = Bundle(default_bundle() if automatic else bundle, download=automatic)
        if self.bundle.preprocessing != RECIPE:
            raise ValueError("Unsupported preprocessing recipe; use the matching deployment runtime")
        self.backend, self.device, self.batch_size, self.threads = backend, str(device), batch_size, threads
        self._sessions, self._torch_model = {}, None
        self._hierarchy_plans = {}
        self.runtime_timings = {}
        self._model_events = []
        self._download_stream = None
        self.onnx_session_info = {}
        self._lock = threading.RLock()
        self.weights = weights
        if weights is not None:
            # This adapter supports the pinned release architecture only; do not accept arbitrary pickles.
            with Path(weights).open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            expected = self.bundle.manifest["files"][self.bundle.manifest["profiles"]["torch"]["model"]]["sha256"]
            if digest != expected:
                raise ValueError("Local weights must match the pinned release checkpoint")
        self.preset = self._preset_name(model or "full")
        if class_list is not None:
            self._select_labels(self._read_list(class_list))
            self.preset = "custom"
        else:
            labels = (
                self.bundle.classes["labels"][0]
                if self.preset == "full"
                else self.bundle.file(self.bundle.regions[self.preset]["path"]).read_text().splitlines()
            )
            self._select_labels(labels)
        if class_mask is not None:
            self._apply_class_mask(class_mask)

    def _preset_name(self, value):
        def normalize(text):
            return re.sub(r"[^a-z0-9]", "", text.lower())

        names = list(self.bundle.regions) + ["full"]
        query = normalize(str(value))
        exact = [name for name in names if normalize(name) == query]
        matches = exact or [name for name in names if query and normalize(name).startswith(query)]
        if len(matches) != 1:
            raise ValueError(f"Unknown or ambiguous preset {value!r}; choose from {', '.join(names)}")
        return matches[0]

    @staticmethod
    def _read_list(value):
        if isinstance(value, (str, Path)):
            return Path(value).read_text().splitlines()
        return list(value)

    def _select_labels(self, labels):
        requested = {str(label).strip() for label in labels if str(label).strip()}
        vocabulary = self.bundle.classes["labels"][0]
        unknown = requested - set(vocabulary)
        if unknown:
            raise ValueError(f"Unknown species IDs: {', '.join(sorted(unknown)[:20])}")
        if not requested:
            raise ValueError("Class list is empty")
        self.selected = np.array([i for i, label in enumerate(vocabulary) if label in requested], dtype=np.int64)
        self.class_list = [vocabulary[i] for i in self.selected]
        self.class_list_sha256 = hashlib.sha256(("\n".join(self.class_list) + "\n").encode()).hexdigest()

    def _apply_class_mask(self, mask):
        with self._lock:
            if mask is None or isinstance(mask, int) and mask == -1:
                self._select_labels(self.bundle.classes["labels"][0])
                self.preset = "full"
                return
            if isinstance(mask, int):
                raise ValueError("Only -1 is a valid integer mask flag")
            if hasattr(mask, "detach"):
                mask = mask.detach().cpu().numpy()
            mask = list(mask)
            if not mask:
                raise ValueError("Class mask is empty")
            if all(isinstance(value, (bool, np.bool_)) for value in mask):
                if len(mask) != len(self.bundle.classes["labels"][0]):
                    raise ValueError("Boolean class mask must cover the full species vocabulary")
                mask = [i for i, keep in enumerate(mask) if keep]
            labels = []
            for value in mask:
                if isinstance(value, str):
                    labels.append(value)
                elif isinstance(value, (int, np.integer)) and 0 <= value < len(self.bundle.classes["labels"][0]):
                    labels.append(self.bundle.classes["labels"][0][value])
                else:
                    raise ValueError(f"Invalid class-mask value: {value!r}")
            self._select_labels(labels)
            self.preset = "custom"

    def available_presets(self):
        return {"full": {"count": len(self.bundle.classes["labels"][0]), "scope": "All model species"}, **self.bundle.regions}

    @cached_property
    def _onnx_api(self):
        try:
            import onnxruntime as ort
        except ImportError as error:
            raise ImportError("Install mambo-v3[onnx], or onnxruntime-gpu for CUDA") from error
        return ort

    @cached_property
    def _torch_api(self):
        try:
            import torch

            from mini_trainer.modeling.classifier import bypass_submodule
        except ImportError as error:
            raise ImportError("Install the matching mini_trainer wheel and a suitable PyTorch backend") from error
        return torch, bypass_submodule

    @cached_property
    def _decode(self):
        return TorchDecode(self._torch_api[0]) if self.backend == "torch" else _rgb

    @cached_property
    def _device_preprocess(self):
        return TorchPreprocess(self._torch_api[0], self.device)

    @property
    def _compact_inputs(self):
        return self.backend == "torch" and self.device != "cpu"

    def _onnx(self, images, embeddings):
        ort = self._onnx_api
        key = "onnx-embedding" if embeddings else "onnx"
        if key not in self._sessions:
            path = self.bundle.profile(key)
            if self.device == "cpu":
                providers = ["CPUExecutionProvider"]
            else:
                if "CUDAExecutionProvider" not in ort.get_available_providers():
                    raise RuntimeError("ONNX CUDA is unavailable; install onnxruntime-gpu and its CUDA/cuDNN dependencies")
                if hasattr(ort, "preload_dlls"):
                    ort.preload_dlls()
                providers = [
                    (
                        "CUDAExecutionProvider",
                        {
                            "device_id": int(self.device.split(":")[-1]) if ":" in self.device else 0,
                            "use_tf32": int(self.effective_precision == "tf32"),
                        },
                    ),
                    "CPUExecutionProvider",
                ]
            session, info = create_session(ort, path, providers, self.threads, cuda=self.device != "cpu", embeddings=embeddings)
            self.onnx_session_info[key] = info
            self._sessions[key] = session
        outputs = ["output_0", "embedding"] if embeddings else ["output_0"]
        if isinstance(images, ort.OrtValue):
            binding = self._sessions[key].io_binding()
            binding.bind_ortvalue_input("images", images)
            for name in outputs:
                binding.bind_output(name, "cpu")
            self._sessions[key].run_with_iobinding(binding)
            values = [value.numpy() for value in binding.get_outputs()]
        else:
            values = self._sessions[key].run(outputs, {"images": images})
        return values[0], values[1] if embeddings else None

    def _torch(self, images, embeddings, *, tensors=False):
        torch, bypass_submodule = self._torch_api
        if self._torch_model is None:
            from mini_trainer.builders import BaseBuilder

            path = self.weights or self.bundle.profile("torch")
            if self.device != "cpu" and not torch.cuda.is_available():
                raise RuntimeError("PyTorch CUDA is unavailable; explicitly choose device='cpu' or install/configure CUDA")
            if self.effective_precision == "bf16":
                with torch.cuda.device(self.device):
                    if not torch.cuda.is_bf16_supported(including_emulation=False):
                        raise RuntimeError("BF16 requires native device support; choose fp16 or fp32")
            # Full local state and pretrained=False prevent constructor downloads.
            state = torch.load(str(path), map_location="cpu", weights_only=True)
            self._torch_model, _ = BaseBuilder.build_model(
                weights=state, device="cpu", dtype=torch.float32, model_args={"pretrained": False}
            )
            self._torch_model.to(self.device).eval()
        head = self._torch_model.classifier
        device_type = self.device.split(":")[0]
        amp = self.effective_precision in ("fp16", "bf16")
        dtype = torch.bfloat16 if self.effective_precision == "bf16" else torch.float16
        with torch.inference_mode():
            # Use the existing backbone boundary; keep the complete head in FP32.
            with (
                torch.autocast(device_type, dtype=dtype, enabled=amp),
                bypass_submodule(self._torch_model, self._torch_model._backbone_output_name),
            ):
                tensor = images if isinstance(images, torch.Tensor) else torch.from_numpy(images).to(self.device)
                if tensor.dtype == torch.uint8:
                    tensor = self._device_preprocess(tensor)
                events = None
                if tensors and tensor.device.type == "cuda":
                    events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                    events[0].record(torch.cuda.current_stream(tensor.device))
                features = self._torch_model(tensor)
            with torch.autocast(device_type, enabled=False):
                features = features.float()
                output = head(features)
                embedding = head.preclassification(features) if embeddings else None
                if events is not None:
                    events[1].record(torch.cuda.current_stream(tensor.device))
                    self._model_events.append(events)
                if embedding is not None and not tensors:
                    embedding = embedding.cpu().numpy()
                return (output if tensors else output[0].float().cpu().numpy()), embedding

    def hierarchy_plan(self, selected):
        # Hash contiguous bytes in native code, not one Python integer per species.
        key = np.asarray(selected, dtype=np.int64).tobytes()
        plan = self._hierarchy_plans.get(key)
        if plan is None:
            plan = self._hierarchy_plans[key] = HierarchyPlan(selected, self.bundle.classes)
        return plan

    def _ranked_views(self, views, view_count, selectors, embeddings=False, *, defer=False):
        """Keep native ranks on Torch; reduce masked/averaged leaves on the same device."""
        if self.backend != "torch":
            leaf, vectors = (
                infer_prepared(self._infer, views, view_count, embeddings)
                if self.tta is not None
                else self._infer(next(iter(views)), embeddings)
            )
            return (lambda: (leaf, vectors, None)) if defer else (leaf, vectors, None)
        torch, _ = self._torch_api

        leaf, vectors, output, norms = None, None, None, None
        self._model_events.clear()
        with torch.inference_mode():
            for view in views:
                output, embedding = self._torch(view, embeddings, tensors=True)
                part = output[0].float() / view_count if self.tta is not None else output[0].float()
                leaf = part if leaf is None else leaf + part
                if embeddings:
                    embedding = torch.as_tensor(embedding, device=leaf.device).float()
                    part = embedding / view_count if self.tta is not None else embedding
                    vectors = part if vectors is None else vectors + part
            if embeddings and self.tta is not None:
                norms = torch.linalg.vector_norm(vectors, dim=1, keepdim=True)
                vectors /= norms
            native = output if self.tta is None else None
            plans = {name: self.hierarchy_plan(selected) for name, selected in selectors.items()}
            ranks = {name: plan.torch_values(leaf, native) for name, plan in plans.items()}
            tensors = [value for raw, _, _ in ranks.values() for value in raw]
            full_name = next((name for name, plan in plans.items() if plan.full), None)
            if full_name is None:
                tensors.append(leaf)
            if vectors is not None:
                tensors.append(vectors)
            if norms is not None:
                tensors.append(norms)
            if defer and leaf.device.type == "cuda" and self._download_stream is None:
                self._download_stream = torch.cuda.Stream(device=leaf.device)
            download = download_tensors(
                tensors, self.runtime_timings, torch=torch, defer=True, stream=self._download_stream if defer else None
            )
            events = tuple(self._model_events)
            self._model_events.clear()

        def finish():
            downloaded = iter(download())
            for begin, end in events:
                self.runtime_timings["model_stream_seconds"] = (
                    self.runtime_timings.get("model_stream_seconds", 0.0) + begin.elapsed_time(end) / 1000
                )
            completed = {name: ([next(downloaded) for _ in raw], labels, mapping) for name, (raw, labels, mapping) in ranks.items()}
            leaf_array = completed[full_name][0][0] if full_name is not None else next(downloaded)
            embedding_array = next(downloaded) if vectors is not None else None
            if norms is not None:
                norm_array = next(downloaded)
                if not np.isfinite(norm_array).all() or np.any(norm_array <= np.finfo(np.float32).eps):
                    raise RuntimeError("TTA produced an undefined mean embedding")
            return leaf_array, embedding_array, completed

        return finish if defer else finish()

    def prepared_batches(self, items, batch_size, *, device_prefetch=True, stats=None, **options):
        """Yield backend inputs: CUDA Torch uint8 squares, otherwise normalized FP32.

        Device slots remain valid until the next iteration.
        """
        stats = {} if stats is None else stats
        accelerated = device_prefetch and self.device != "cpu"
        compact = self._compact_inputs
        factory = pinned_factory(self.device, compact=compact) if accelerated and self.backend == "torch" else None
        source = prepared_stream(
            items,
            batch_size,
            tta=self.tta,
            stats=stats,
            reuse_buffers=True,
            buffer_factory=factory,
            compact=compact,
            decode=self._decode,
            **options,
        )
        if accelerated:
            yield from device_batches(source, self.backend, self.device, stats)
        else:
            with closing(source):
                for offset, views in source:
                    yield offset, views, len(views[0])

    def _prepare(self, batch, pool=None):
        return prepare_batch(batch, pool, compact=self._compact_inputs, decode=self._decode)

    def _infer(self, images, embeddings=False):
        return (self._torch if self.backend == "torch" else self._onnx)(images, embeddings)

    def _infer_batch(self, batch, embeddings=False, pool=None):
        if self.tta is not None:
            return infer_augmented(self._infer, batch, self.tta, embeddings, pool)
        return self._infer(self._prepare(batch, pool), embeddings)

    def _predict(self, x, embeddings=False, topk=1):
        with self._lock:
            leaf_batches, embedding_batches, rank_batches = [], [], []
            items = image_items(x)
            with ThreadPoolExecutor(max_workers=self.preprocess_workers) if self.preprocess_workers > 1 else nullcontext(None) as pool:
                while batch := list(islice(items, self.batch_size)):
                    if self.backend == "torch":
                        views = (
                            prepared_views(batch, self.tta, pool, compact=self._compact_inputs, decode=self._decode)
                            if self.tta
                            else (self._prepare(batch, pool),)
                        )
                        leaves, vectors, ranks = self._ranked_views(
                            views, len(self.tta.transforms) if self.tta else 1, {"selected": self.selected}, embeddings
                        )
                        rank_batches.append(ranks["selected"])
                    else:
                        leaves, vectors = self._infer_batch(batch, embeddings, pool)
                    if not np.isfinite(leaves).all():
                        raise RuntimeError("Model returned non-finite species scores")
                    leaf_batches.append(leaves)
                    if embeddings:
                        embedding_batches.append(vectors)
            if not leaf_batches:
                raise ValueError("No images supplied")
            if rank_batches:
                raw = [np.concatenate([batch[0][rank] for batch in rank_batches]) for rank in range(len(rank_batches[0][0]))]
                labels, mappings = rank_batches[0][1:]
            else:
                raw, labels, mappings = self.hierarchy_plan(self.selected).numpy(np.concatenate(leaf_batches))
            result = Prediction(
                raw,
                labels,
                mappings,
                topk,
                **self._prediction_metadata(),
            )
            return (result, np.concatenate(embedding_batches)) if embeddings else result

    def _prediction_metadata(self):
        return dict(
            model_id=self.bundle.manifest["model_id"],
            artifact_revision=self.bundle.manifest.get("artifact_revision"),
            bundle_sha256=self.bundle.manifest_sha256,
            preprocessing_id=self.bundle.preprocessing["id"],
            backend=self.backend,
            preset=self.preset,
            class_list_sha256=self.class_list_sha256,
            precision=self.effective_precision,
            tta=self.tta.name if self.tta else "none",
            tta_views=len(self.tta.transforms) if self.tta else 1,
        )

    def predict(self, x, topk=1):
        return self._predict(x, topk=topk)

    def __call__(self, x, **kwargs):
        return self.predict(x, **kwargs)

    def predict_with_embeddings(self, x, topk=1):
        return self._predict(x, embeddings=True, topk=topk)

    def predict_stream(
        self,
        paths,
        *,
        embeddings=False,
        topk=1,
        read_workers=32,
        prepare_workers=None,
        read_window=None,
        prefetch_batches=2,
        encoded_budget=256 * 1024**2,
        stats=None,
        device_prefetch=True,
    ):
        """Yield one Prediction (or Prediction/embedding pair) per batch of paths.

        Use contextlib.closing when stopping before exhaustion. Input paths are lazy;
        outputs are not accumulated. Loading settings are explicit per stream.
        """
        batches = self.prepared_batches(
            ((path, None) for path in paths),
            self.batch_size,
            device_prefetch=device_prefetch,
            read_workers=read_workers,
            prepare_workers=self.preprocess_workers if prepare_workers is None else prepare_workers,
            read_window=max(128, self.batch_size) if read_window is None else read_window,
            prefetch_batches=prefetch_batches,
            encoded_budget=encoded_budget,
            stats=stats,
        )

        def process(resolve, plan, metadata):
            leaves, vectors, ranks = resolve()
            if not np.isfinite(leaves).all():
                raise RuntimeError("Model returned non-finite species scores")
            result = Prediction(*(ranks["selected"] if ranks is not None else plan.numpy(leaves)), topk, **metadata)
            return (result, vectors) if embeddings else result

        with closing(batches), ResultWorker(process) as worker:
            for _, views, _ in batches:
                with self._lock:
                    resolve = self._ranked_views(views, len(views), {"selected": self.selected}, embeddings, defer=True)
                    worker.submit(
                        resolve,
                        self.hierarchy_plan(self.selected),
                        self._prediction_metadata(),
                    )
                if len(worker.pending) == 2:
                    yield worker.pop()
            while worker.pending:
                yield worker.pop()
