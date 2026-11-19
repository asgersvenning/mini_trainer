"""Releases: a checkpoint as a trunk and heads (safetensors and ONNX), verified, packaged for Hugging Face.

See ``docs/release.md``. ``standalone`` is shipped inside each release as ``predict.py`` and imports
nothing from mini_trainer.
"""

from .build import add_head, export_release
from .card import package_hf
from .verify import verify_release

__all__ = ["add_head", "export_release", "package_hf", "verify_release"]
