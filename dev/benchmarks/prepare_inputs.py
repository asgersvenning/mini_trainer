"""Compatibility factory path retained by historical preprocessing recipes.

Run new preparation commands through dev.benchmarks.inference.prepare_inputs.
"""

from dev.benchmarks.inference.prepare_inputs import repository_preprocess

__all__ = ["repository_preprocess"]
