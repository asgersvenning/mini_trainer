import argparse
import collections
import inspect
import json
import sys
import time
from pathlib import Path

from deployment.mambo_deploy.preprocessing import preprocess
from dev.releases.mambo_v3.evaluate import prepare_batch

parser = argparse.ArgumentParser(description="Line attribution for the current release preprocessor")
parser.add_argument("--evidence", type=Path, required=True)
parser.add_argument("--root", type=Path, required=True)
args = parser.parse_args()
root = args.evidence
records = json.loads((root / "samples.json").read_text())
paths = [args.root / r["path"] for r in records]
line_times = collections.defaultdict(float)
state = {}
layout = {}


def trace(frame, event, arg):
    if frame.f_code is not preprocess.__code__:
        return
    now = time.perf_counter()
    if event in ("line", "return"):
        if "line" in state:
            line_times[state["line"]] += now - state["t"]
        state.update(line=frame.f_lineno, t=now)
    if event == "return":
        for name in ("coordinates", "lo", "fraction", "image", "rows", "pixels"):
            value = frame.f_locals[name]
            layout[name] = dict(
                dtype=str(value.dtype), shape=list(value.shape), strides=list(value.strides), contiguous=value.flags.c_contiguous
            )
        state.clear()
    return trace


sys.settrace(trace)
prepare_batch(paths)
sys.settrace(None)
source = Path(inspect.getfile(preprocess)).read_text().splitlines()
rows = [
    {"line": line, "code": source[line - 1].strip(), "ms": seconds * 1000}
    for line, seconds in sorted(line_times.items(), key=lambda i: -i[1])
]
print(json.dumps(rows, indent=2))
(root / "preprocess-line-profile.json").write_text(json.dumps(rows, indent=2))

(root / "preprocess-layout.json").write_text(json.dumps(layout, indent=2))
