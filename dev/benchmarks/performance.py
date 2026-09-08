"""Preserve benchmark timing and memory across the logger's phase resets."""

import json
import time
from pathlib import Path

import torch

from mini_trainer.logging import MultiLogger


class BenchmarkLogger(MultiLogger):
    def __init__(self, *args, measurement_device="cpu", **kwargs):
        self.measurement_device = torch.device(measurement_device)
        self.peak_allocated_bytes = 0
        self.phase_measurements = []
        self.phase_active = False
        self.phase_peak_allocated_bytes = 0
        super().__init__(*args, **kwargs)

    def _reset_cuda_memory_stats(self):
        if self.measurement_device.type == "cuda":
            peak = torch.cuda.max_memory_allocated(self.measurement_device)
            self.peak_allocated_bytes = max(self.peak_allocated_bytes, peak)
            if self.phase_active:
                self.phase_peak_allocated_bytes = max(self.phase_peak_allocated_bytes, peak)
        super()._reset_cuda_memory_stats()

    def _synchronize(self):
        if self.measurement_device.type == "cuda":
            torch.cuda.synchronize(self.measurement_device)

    def start_timing(self):
        self._synchronize()
        super().start_timing()
        self.phase_started = time.perf_counter()
        self.phase_active = True
        self.phase_peak_allocated_bytes = 0

    def stop_timing(self):
        self._synchronize()
        self.phase_measurements.append(
            {
                "epoch": self._epoch,
                "phase": self._type,
                "seconds": time.perf_counter() - self.phase_started,
                "peak_cuda_allocated_bytes": (
                    max(self.phase_peak_allocated_bytes, torch.cuda.max_memory_allocated(self.measurement_device))
                    if self.measurement_device.type == "cuda"
                    else None
                ),
            }
        )
        self.phase_active = False
        super().stop_timing()

    def finish(self):
        super().finish()  # Captures the last peak before the final reset.
        if self.output_dir is not None:
            Path(self.output_dir, "performance.json").write_text(
                json.dumps(
                    {
                        "phases": self.phase_measurements,
                        "peak_cuda_allocated_bytes": self.peak_allocated_bytes if self.measurement_device.type == "cuda" else None,
                        "phase_scope": (
                            "synchronized batch loop including loader, preprocessing, compute and batch logging; "
                            "excludes figures and checkpoints"
                        ),
                    },
                    indent=2,
                )
                + "\n"
            )
