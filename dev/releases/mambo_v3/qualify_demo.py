"""Check the staged single-image UI with installed runtimes; never launch a server."""

import argparse
import json
import runpy
from pathlib import Path

from PIL import Image


def qualify(app_path, image_path, output):
    demo = runpy.run_path(str(app_path))
    interface = demo["build_app"]()
    if not interface.blocks:
        raise AssertionError("Empty demo interface")
    records = []
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    for backend in ("onnx", "torch"):
        predictor = demo["predictor_for"](backend)
        custom = predictor.class_list[0]
        runtime = None
        for preset, labels, tta in (("full", None, False), ("north_europe", "", True), ("full", custom, False)):
            tables = demo["classify"](image, backend, preset, labels, tta, 5)
            if not all(table and all(0 <= row[3] <= 100 and row[2] for row in table) for table in tables[:3]):
                raise AssertionError("Missing predictions, taxon IDs or valid confidences")
            current = predictor._torch_model if backend == "torch" else predictor._sessions["onnx"]
            if runtime is not None and current is not runtime:
                raise AssertionError("Changing demo configuration reloaded the runtime")
            runtime = current
            records.append(
                {
                    "backend": backend,
                    "preset": preset,
                    "custom": bool(labels),
                    "tta": tta,
                    "rows": [len(table) for table in tables[:3]],
                    "status": tables[-1],
                }
            )
    if demo["predictor_for"].cache_info().currsize != 2:
        raise AssertionError("Unexpected runtime cache size")
    output.write_text(json.dumps({"controls": records, "runtime_reuse": True, "ui_constructed": True}, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("app", type=Path)
    parser.add_argument("image", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    qualify(args.app, args.image, args.output)
