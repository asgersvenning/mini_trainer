"""Two explicit backends sharing one image, vocabulary and result contract."""

import hashlib
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from itertools import islice
from pathlib import Path

import numpy as np

from .augmentation import infer_augmented, resolve_tta
from .bundle import Bundle
from .download import default_bundle
from .onnx_session import create_session
from .preprocessing import RECIPE, image_items, preprocess
from .results import Prediction, hierarchy


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
        self.preset = self._preset_name(model or ("full" if weights is not None else "europe"))
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

    def _onnx(self, images, embeddings):
        try:
            import onnxruntime as ort
        except ImportError as error:
            raise ImportError("Install mambo-deploy[onnx], or onnxruntime-gpu for CUDA") from error
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
        values = self._sessions[key].run(outputs, {"images": images})
        return values[0], values[1] if embeddings else None

    def _torch(self, images, embeddings):
        try:
            import torch

            from mini_trainer.builders import BaseBuilder
            from mini_trainer.modeling.classifier import bypass_submodule
        except ImportError as error:
            raise ImportError("Install the matching mini_trainer wheel and a suitable PyTorch backend") from error
        if self._torch_model is None:
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
                features = self._torch_model(torch.from_numpy(images).to(self.device))
            with torch.autocast(device_type, enabled=False):
                features = features.float()
                output = head(features)
                embedding = head.preclassification(features).cpu().numpy() if embeddings else None
                return output[0].float().cpu().numpy(), embedding

    def _prepare(self, batch, pool=None):
        return np.stack(list(pool.map(preprocess, batch)) if pool and len(batch) > 1 else [preprocess(item) for item in batch])

    def _infer(self, images, embeddings=False):
        return (self._torch if self.backend == "torch" else self._onnx)(images, embeddings)

    def _infer_batch(self, batch, embeddings=False, pool=None):
        if self.tta is not None:
            return infer_augmented(self._infer, batch, self.tta, embeddings, pool)
        return self._infer(self._prepare(batch, pool), embeddings)

    def _predict(self, x, embeddings=False, topk=1):
        with self._lock:
            leaf_batches, embedding_batches = [], []
            items = image_items(x)
            with ThreadPoolExecutor(max_workers=self.preprocess_workers) if self.preprocess_workers > 1 else nullcontext(None) as pool:
                while batch := list(islice(items, self.batch_size)):
                    leaves, vectors = self._infer_batch(batch, embeddings, pool)
                    if not np.isfinite(leaves).all():
                        raise RuntimeError("Model returned non-finite species scores")
                    leaf_batches.append(leaves)
                    if embeddings:
                        embedding_batches.append(vectors)
            if not leaf_batches:
                raise ValueError("No images supplied")
            raw, labels, mappings = hierarchy(np.concatenate(leaf_batches), self.selected, self.bundle.classes)
            result = Prediction(
                raw,
                labels,
                mappings,
                topk,
                model_id=self.bundle.manifest["model_id"],
                backend=self.backend,
                preset=self.preset,
                class_list_sha256=self.class_list_sha256,
                precision=self.effective_precision,
                tta=self.tta.name if self.tta else "none",
                tta_views=len(self.tta.transforms) if self.tta else 1,
            )
            return (result, np.concatenate(embedding_batches)) if embeddings else result

    def predict(self, x, topk=1):
        return self._predict(x, topk=topk)

    def __call__(self, x, **kwargs):
        return self.predict(x, **kwargs)

    def predict_with_embeddings(self, x, topk=1):
        return self._predict(x, embeddings=True, topk=topk)
