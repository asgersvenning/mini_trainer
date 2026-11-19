"""Verify a release on real images against mini_trainer's own evaluation path.

The reference is the checkpoint as ``mt_predict`` runs it: the checkpoint's reader and transform, the
trunk's features (classifier bypassed, as :func:`mini_trainer.modeling.predict`), then the classifier,
whose embedding is collected through ``EmbeddingContext``. Every artifact is then run the way a user
would run it, through the shipped ``standalone`` module (``predict.py``): its own image reader, each
trunk graph, and the head scored from its safetensors file. Results go to ``verification.json``.
"""

import gc
import json
import time
from pathlib import Path

import numpy as np
import torch

from mini_trainer.modeling import EmbeddingContext

from . import standalone
from .build import load_checkpoint, split

# Gates. fp32 artifacts differ from PyTorch by float rounding only; fp16 trunks by half-precision rounding.
GATES = {"fp32": {"top1": 0.99, "cosine": 0.9999}, "fp16": {"top1": 0.95, "cosine": 0.999}}


def batches(items, size):
    for start in range(0, len(items), size):
        yield items[start : start + size]


@torch.inference_mode()
def reference(weights, paths, batch_size=16) -> dict:
    """mini_trainer's outputs: pixels as its reader returns them, features, embedding, logits per rank."""
    model, preprocess = load_checkpoint(weights)
    trunk, classifier = split(model)
    out = {"pixels": [], "features": [], "embedding": [], "logits": []}
    for chunk in batches(paths, batch_size):
        pixels = torch.stack([preprocess.reader(path) for path in chunk])
        features = trunk(torch.stack([preprocess(image) for image in pixels]))
        with EmbeddingContext():
            logits = classifier(features)
            embedding = EmbeddingContext.get()
        logits = logits if isinstance(logits, (list, tuple)) else [logits]
        out["pixels"].append(pixels.numpy())
        out["features"].append(features.numpy())
        out["embedding"].append(embedding.numpy())
        out["logits"].append([z.numpy() for z in logits])
    return {
        "pixels": np.concatenate(out["pixels"]),
        "features": np.concatenate(out["features"]),
        "embedding": np.concatenate(out["embedding"]),
        "logits": [np.concatenate(ranks) for ranks in zip(*out["logits"])],
    }


def compare(ref: dict, features, embedding, logits, ranks) -> dict:
    """Per-rank top-1 agreement and logit differences; feature and embedding agreement."""
    cosine = (embedding * ref["embedding"]).sum(1) / np.linalg.norm(embedding, axis=1) / np.linalg.norm(ref["embedding"], axis=1)
    result = {"embedding_cosine_min": float(cosine.min()), "embedding_cosine_mean": float(cosine.mean())}
    if features is not None:
        scale = np.abs(ref["features"]).max()
        result["features_max_abs_diff_relative"] = float(np.abs(features - ref["features"]).max() / scale)
    result["ranks"] = {}
    for name, got, want in zip(ranks, logits, ref["logits"]):
        finite = np.isfinite(want)
        if not np.array_equal(finite, np.isfinite(got)):
            raise AssertionError(f"{name}: empty nodes (-inf) differ")
        diff = np.abs(got[finite] - want[finite])
        result["ranks"][name] = {
            "top1_agreement": float((got.argmax(1) == want.argmax(1)).mean()),
            "max_abs_logit_diff": float(diff.max()),
            "mean_abs_logit_diff": float(diff.mean()),
        }
    return result


def passes(result: dict, precision: str) -> bool:
    gate = GATES[precision]
    agreement = min(rank["top1_agreement"] for rank in result["ranks"].values())
    return agreement >= gate["top1"] and result["embedding_cosine_min"] >= gate["cosine"]


def run_trunk(session, pixels, batch_size=16):
    out = [session.run(["features", "embedding"], {"images": chunk}) for chunk in batches(pixels, batch_size)]
    return np.concatenate([f for f, _ in out]), np.concatenate([e for _, e in out])


def score_all(features, head, batch_size=16):
    out = [standalone.score(chunk, head) for chunk in batches(features, batch_size)]
    return np.concatenate([e for e, _ in out]), [np.concatenate(r) for r in zip(*[ranks for _, ranks in out])]


def latency(fn, pixels, repeats=10) -> dict:
    """Median milliseconds for one image, and images per second at batch 16."""
    one, many = pixels[:1], pixels[:16]
    fn(one)
    fn(many)
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn(one)
        times.append(time.perf_counter() - start)
    start = time.perf_counter()
    fn(many)
    return {"ms_per_image_batch1": 1000 * float(np.median(times)), "images_per_s_batch16": len(many) / (time.perf_counter() - start)}


def tensor_ranges(path: Path) -> dict[str, tuple]:
    """A safetensors file's tensors: name -> (dtype, shape, start, end) byte offsets in the file."""
    with path.open("rb") as handle:
        size = int.from_bytes(handle.read(8), "little")
        header = json.loads(handle.read(size))
    header.pop("__metadata__", None)
    return {
        name: (t["dtype"], tuple(t["shape"]), 8 + size + t["data_offsets"][0], 8 + size + t["data_offsets"][1])
        for name, t in header.items()
    }


def same_tensors(folder: Path, combined: str, parts: dict[str, str], chunk: int = 1 << 24) -> bool:
    """Whether ``combined`` holds exactly the tensors of its part files (renamed with a prefix), byte for byte.

    Reads the files in chunks, so a multi-gigabyte release is compared in a few megabytes of memory.
    """
    whole = tensor_ranges(folder / combined)
    expected = {}
    for prefix, file in parts.items():
        expected.update({prefix + name: (file, entry) for name, entry in tensor_ranges(folder / file).items()})
    if set(whole) != set(expected):
        return False
    with (folder / combined).open("rb") as a:
        for name, (file, (dtype, shape, start, end)) in expected.items():
            other = whole[name]
            if other[:2] != (dtype, shape) or other[3] - other[2] != end - start:
                return False
            with (folder / file).open("rb") as b:
                a.seek(other[2])
                b.seek(start)
                for offset in range(0, end - start, chunk):
                    n = min(chunk, end - start - offset)
                    if a.read(n) != b.read(n):
                        return False
    return True


def verify_release(release, weights, paths, head=None, threads=4, timing=True) -> dict:
    """Run every artifact of ``release`` on ``paths``; write and return ``verification.json``'s content."""
    folder = Path(release)
    config = json.loads((folder / "config.json").read_text())
    name = head or config["default_head"]
    spec = config["heads"][name]
    paths = [str(p) for p in paths]
    print(f"reference: mini_trainer on {len(paths)} images", flush=True)
    ref = reference(weights, paths)
    gc.collect()  # release the checkpoint's model before the ONNX sessions load

    pixels = np.stack([standalone.read_square(path, config["preprocessing"]["read_size"]) for path in paths])
    same = (pixels == ref["pixels"]).all(axis=(1, 2, 3))
    report = {
        "images": len(paths),
        "example_images": paths[:3],
        "head": name,
        "threads": threads,
        "preprocessing": {
            "identical_images": float(same.mean()),
            "max_abs_pixel_diff": int(np.abs(pixels.astype(int) - ref["pixels"].astype(int)).max()),
        },
        "artifacts": {},
    }
    loaded = standalone.load_head(folder / spec["file"], folder / spec["labels"])
    ranks = loaded.ranks
    trunks = [("trunk.onnx", "fp32", config["trunk"]["onnx"])]
    if "onnx_fp16" in config["trunk"]:
        trunks.append(("trunk_fp16.onnx", "fp16", config["trunk"]["onnx_fp16"]))
    for label, precision, file in trunks:
        print(f"{label} + heads/{name}.safetensors", flush=True)
        session = standalone.onnx_session(folder / file, threads)
        features, _ = run_trunk(session, pixels)
        embedding, logits = score_all(features, loaded)
        result = compare(ref, features, embedding, logits, ranks)
        if timing:
            result["trunk_latency"] = latency(lambda x: session.run(["features"], {"images": x}), pixels)
        report["artifacts"][label] = {"precision": precision, **result, "pass": passes(result, precision)}
        del session
    if timing:
        report["head_latency"] = latency(lambda x: standalone.score(ref["features"][: len(x)], loaded), pixels)

    if name == config["default_head"] and "combined" in config:
        print("model.onnx (combined)", flush=True)
        session = standalone.onnx_session(folder / "model.onnx", threads)
        outputs = [session.run(None, {"images": chunk}) for chunk in batches(pixels, 16)]
        embedding = np.concatenate([o[0] for o in outputs])
        logits = [np.concatenate([o[k] for o in outputs]) for k in range(1, len(ranks) + 1)]
        result = compare(ref, None, embedding, logits, ranks)
        if timing:
            result["latency"] = latency(lambda x: session.run(None, {"images": x}), pixels)
        report["artifacts"]["model.onnx"] = {"precision": "fp32", **result, "pass": passes(result, "fp32")}
        del session

    try:
        trunk = standalone.torch_trunk(folder)
    except (ImportError, ValueError) as error:
        report["artifacts"]["trunk.safetensors"] = {"skipped": str(error)}
    else:
        print("trunk.safetensors through timm", flush=True)
        torch.set_num_threads(threads)
        with torch.inference_mode():
            features = np.concatenate([trunk(torch.from_numpy(chunk)).numpy() for chunk in batches(pixels, 16)])
        embedding, logits = score_all(features, loaded)
        result = compare(ref, features, embedding, logits, ranks)
        report["artifacts"]["trunk.safetensors"] = {"precision": "fp32", **result, "pass": passes(result, "fp32")}
        del trunk

    if (folder / "model.safetensors").exists():
        parts = {"trunk.": config["trunk"]["safetensors"], "head.": config["heads"][config["default_head"]]["file"]}
        identical = same_tensors(folder, "model.safetensors", parts)
        report["artifacts"]["model.safetensors"] = {"identical_to_separate_files": identical, "pass": identical}

    report["sizes_mb"] = {path: round(entry["size"] / 1e6, 1) for path, entry in config["files"].items()}
    report["pass"] = bool(same.all()) and all(a.get("pass", True) for a in report["artifacts"].values())
    (folder / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def summary(report: dict) -> str:
    """A short table of the verification report."""
    lines = [f"{report['images']} images; preprocessing identical for {report['preprocessing']['identical_images']:.1%}"]
    for name, art in report["artifacts"].items():
        if "ranks" not in art:
            detail = art.get("skipped") or ("identical to the separate files" if art.get("pass") else "differs from the separate files")
            lines.append(f"  {name:18s} {'PASS' if art.get('pass') else 'FAIL' if 'pass' in art else 'SKIP'}  {detail}")
            continue
        worst = min(art["ranks"].values(), key=lambda r: r["top1_agreement"])
        dmax = max(r["max_abs_logit_diff"] for r in art["ranks"].values())
        lines.append(
            f"  {name:18s} {'PASS' if art['pass'] else 'FAIL'}  top-1 agreement (worst rank) {worst['top1_agreement']:.4f}"
            f"  max |dlogit| {dmax:.2e}  embedding cosine min {art['embedding_cosine_min']:.6f}"
        )
    lines.append("verification " + ("PASSED" if report["pass"] else "FAILED"))
    return "\n".join(lines)
