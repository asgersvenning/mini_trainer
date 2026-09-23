import argparse
import collections
import json
import statistics
import time
from pathlib import Path

import onnxruntime as ort

from deployment.mambo_deploy import Predictor
from dev.releases.mambo_v3.benchmark import snapshot
from dev.releases.mambo_v3.evaluate import prepare_batch

parser = argparse.ArgumentParser(description="ONNX placement and batch-scaling trace")
for name in ("evidence", "root", "bundle"):
    parser.add_argument("--" + name, type=Path, required=True)
args = parser.parse_args()
root = args.evidence
records = json.loads((root / "samples.json").read_text())
paths = [args.root / r["path"] for r in records]
report = {"before": snapshot(), "cells": [], "providers": []}
for n in (1, 8, 32):
    x = prepare_batch(paths[:n])
    p = Predictor(args.bundle, device="cuda:0", model="north_europe", batch_size=32, threads=4)
    # Adapter explicitly disables profiling; construct matching session separately for diagnosis.
    o = ort.SessionOptions()
    o.enable_profiling = True
    o.profile_file_prefix = str(root / f"onnx-{n}")
    o.intra_op_num_threads = 4
    o.inter_op_num_threads = 1
    ort.preload_dlls()
    session = ort.InferenceSession(
        str(p.bundle.profile("onnx")),
        sess_options=o,
        providers=[("CUDAExecutionProvider", {"device_id": 0, "use_tf32": 0}), "CPUExecutionProvider"],
    )
    session.disable_fallback()
    assert session.get_providers()[0] == "CUDAExecutionProvider"
    for _ in range(2):
        session.run(["output_0"], {"images": x})
    vals = []
    for _ in range(7):
        t = time.perf_counter()
        session.run(["output_0"], {"images": x})
        vals.append((time.perf_counter() - t) * 1000)
    trace = Path(session.end_profiling())
    events = json.loads(trace.read_text())
    ops = collections.defaultdict(float)
    counts = collections.Counter()
    for e in events:
        a = e.get("args", {})
        if e.get("cat") == "Node" and "provider" in a:
            key = (a["provider"], a["op_name"])
            ops[key] += e["dur"]
            counts[key] += 1
    report["cells"].append(
        {
            "batch": n,
            "profiled_median_ms": statistics.median(vals),
            "node_totals": [
                {"provider": k[0], "op": k[1], "us": v, "calls": counts[k]} for k, v in sorted(ops.items(), key=lambda i: -i[1])
            ],
            "trace": str(trace),
        }
    )
    print(n, report["cells"][-1], flush=True)
report["after"] = snapshot()
(root / "onnx-profile-summary.json").write_text(json.dumps(report, indent=2))
