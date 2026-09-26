"""Evaluate pinned MAMBO_v2 through its original source in an isolated process."""

import argparse
import csv
import hashlib
import importlib.metadata
import resource
import subprocess
import time
import tomllib
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path

from dev.benchmarks.inference.onnx_inference import file_hash
from dev.releases.mambo_v3.benchmark import snapshot, timing
from dev.releases.mambo_v3.evaluation_data import CSV_COLUMNS, canonical_rows, load_records, write_json

COMMIT = "32b3cd661778356b2e8c4cff5b10fa9061aa6f5d"
PRESETS = ("north_europe", "europe", "full")
FILENAMES = dict(
    zip(PRESETS, ("hierarchical_bioclip2_ft_neu_v1.pt", "hierarchical_bioclip2_ft_eu_v1.pt", "hierarchical_bioclip2_ft_v1.pt"), strict=True)
)


def verify_source(source):
    """Require byte-identical source for every Python file in the pinned release."""
    names = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", COMMIT, "mini_trainer"], text=True).splitlines()
    for name in names:
        if name.endswith(".py"):
            expected = subprocess.check_output(["git", "show", f"{COMMIT}:{name}"])
            if (source / name).read_bytes() != expected:
                raise ValueError(f"Changed legacy source: {name}")
    import mini_trainer

    if Path(mini_trainer.__file__).resolve() != (source / "mini_trainer/__init__.py").resolve():
        raise ValueError("Use python -P with PYTHONPATH=legacy-source:current-checkout")


def load_states(root):
    """Validate all three published masks, and prove learned tensors are identical."""
    import torch

    inventory = tomllib.loads(Path(__file__).with_name("inventory.toml").read_text())
    hashes = {Path(a["path"]).name: a["sha256"] for a in inventory["artifacts"]}
    states = {}
    for preset, filename in FILENAMES.items():
        path = root / filename
        if file_hash(path) != hashes[filename]:
            raise ValueError(f"Changed legacy weights: {path}")
        states[preset] = torch.load(path, map_location="cpu", weights_only=True)
    full = states["full"]
    for state in states.values():
        if state.keys() - {"classifier.active_indices"} != full.keys() - {"classifier.active_indices"}:
            raise ValueError("Legacy state keys differ")
        for key, value in state.items():
            if key == "classifier.active_indices":
                continue
            equal = torch.equal(value, full[key]) if isinstance(value, torch.Tensor) else value == full[key]
            if not equal:
                raise ValueError(f"Learned legacy parameters differ: {key}")
    return states, {name: hashes[filename] for name, filename in FILENAMES.items()}


def verify_backbone():
    """Pin the external frozen backbone as well as the small release head files."""
    from huggingface_hub import hf_hub_download

    expected = {
        "open_clip_config.json": "1bf947e96e943fe50efd5c3e26c37f843a2fa3c358967719a68c8a6d17ce68c8",
        "open_clip_model.safetensors": "b7b2bf6fbc95799e42630e394cf95803892ab447c1a8ab629dbc82fbeaf7dfef",
    }
    for name, digest in expected.items():
        path = hf_hub_download("imageomics/bioclip-2", name, local_files_only=True)
        if file_hash(path) != digest:
            raise ValueError(f"Unexpected cached BioCLIP-2 file: {name}")
    return expected


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "release": "MAMBO_v2", "source_commit": COMMIT, "runner_sha256": file_hash(__file__)}
    try:
        verify_source(args.source)
        import torch
        from mini_trainer.classifier import bypass_submodule, classification_module

        from mini_trainer.deploy import Predictor

        torch.set_num_threads(args.threads)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        report["settings"] = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        report["runtime"] = {
            name: importlib.metadata.version(name) for name in ("torch", "torchvision", "open_clip_torch", "timm", "huggingface_hub")
        }
        report["precision"] = "original API: CUDA float16 autocast; CPU without autocast; checkpoint preprocessing dtype retained; TF32 off"
        report["backbone_sha256"] = verify_backbone()
        states, report["weights_sha256"] = load_states(args.weights)
        masks = {
            name: state["classifier.active_indices"].tolist()
            if "classifier.active_indices" in state
            else list(range(len(state["classifier._extra_state"]["cls2idx"]["0"])))
            for name, state in states.items()
        }
        mapping = states["full"]["classifier._extra_state"]["cls2idx"]["0"]
        inverse = {v: k for k, v in mapping.items()}
        report["lists"] = {
            name: {"count": len(mask), "sha256": hashlib.sha256(("\n".join(inverse[i] for i in mask) + "\n").encode()).hexdigest()}
            for name, mask in masks.items()
        }
        del states
        manifest, records = load_records(
            args.manifest, args.root, max(args.bank_size, max(args.batches or [32])) if args.phase == "benchmark" else args.count
        )
        write_json(args.output / "samples.json", records)
        report.update(
            samples=len(records),
            dataset=manifest["dataset"],
            manifest_sha256=file_hash(args.manifest),
            sample_ids_sha256=file_hash(args.output / "samples.json"),
        )
        report["before"] = snapshot()
        start = time.perf_counter()
        predictor = Predictor(device=args.device, model=str(args.weights / FILENAMES["full"]))
        report["cpu_input_cast"] = args.cpu_float32
        if args.cpu_float32:
            if args.device != "cpu":
                raise ValueError("The ancillary float32 input adapter is CPU-only")
            original_preproc = predictor.preproc
            predictor.preproc = lambda image: original_preproc(image).float()
        first = predictor.predict(str(args.root / records[0]["path"]))
        list(first)  # Complete CPU label/confidence materialization before stopping the clock.
        report["load_and_first_image_seconds"] = time.perf_counter() - start
        classifier = classification_module(predictor.model)
        report["input"] = {
            "reader_size": predictor.resize_size,
            "preprocessed_shape": list(predictor.preproc(predictor.reader(str(args.root / records[0]["path"])).unsqueeze(0)).shape),
        }

        def features(paths, pool):
            with torch.inference_mode(), torch.autocast(device_type=predictor.device.type, enabled=predictor.device.type == "cuda"):
                batch = torch.stack(list(pool.map(predictor.reader, paths)))
                batch = predictor.preproc(batch).to(predictor.device)
                with bypass_submodule(predictor.model, predictor.model._backbone_output_name):
                    return predictor.model(batch)

        if args.phase == "benchmark":
            paths = [str(args.root / r["path"]) for r in records]
            if any(file_hash(path) != r["sha256"] for path, r in zip(paths, records, strict=True)):
                raise ValueError("Benchmark images changed")
            report["cells"] = []
            for size in args.batches or ([1, 8] if args.device == "cpu" else [1, 8, 32]):
                for preset in args.presets:
                    predictor._apply_class_mask(-1 if preset == "full" else masks[preset])

                    def call():
                        prediction = predictor.predict(paths[:size])
                        # Match v3's completed CPU results; no deferred GPU work.
                        prediction.indices.cpu()
                        prediction.confidence.cpu()

                    for _ in range(2):
                        call()
                    observed = timing(call, 7)
                    report["cells"].append({"preset": preset, "batch_size": size, "end_to_end": observed, "resources": snapshot()})
                    print(preset, size, round(size / observed["median_seconds"], 2), flush=True)
        else:
            report["qualification"] = {}
            with ExitStack() as stack:
                pool = stack.enter_context(ThreadPoolExecutor(max_workers=4))
                writers = {}
                for preset in args.presets:
                    directory = args.output / preset
                    directory.mkdir()
                    writer = csv.writer(stack.enter_context((directory / "mini_metric.csv").open("w", newline="")))
                    writer.writerow(CSV_COLUMNS)
                    writers[preset] = writer
                for offset in range(0, len(records), args.batch_size):
                    selected = records[offset : offset + args.batch_size]
                    paths = [str(args.root / r["path"]) for r in selected]
                    if any(file_hash(path) != r["sha256"] for path, r in zip(paths, selected, strict=True)):
                        raise ValueError("Evaluation image changed")
                    encoded = features(paths, pool)
                    with torch.inference_mode(), torch.autocast(device_type=predictor.device.type, enabled=predictor.device.type == "cuda"):
                        for preset in args.presets:
                            predictor._apply_class_mask(-1 if preset == "full" else masks[preset])
                            prediction = classifier.predict(encoded)
                            if args.phase == "qualification" or offset == 0:
                                direct = predictor.predict(paths)
                                if direct.labels != prediction.labels:
                                    raise ValueError(f"Cached-feature predictions differ from original API: {preset}")
                                report["qualification"][preset] = "all sampled labels identical to original API"
                            writers[preset].writerows(canonical_rows(selected, prediction, offset))
                    if offset % (args.batch_size * 50) == 0:
                        print(offset + len(selected), len(records), flush=True)
            report["csv_sha256"] = {name: file_hash(args.output / name / "mini_metric.csv") for name in args.presets}
        if args.device != "cpu":
            report["torch_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            report["torch_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        report["status"] = "complete"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["peak_rss_kib_linux"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report["after"] = snapshot()
        write_json(args.output / "report.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("qualification", "full", "benchmark"))
    for name in ("source", "weights", "manifest", "root", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--count", type=int)
    parser.add_argument("--cpu-float32", action="store_true", help="Ancillary CPU run: cast original preprocessor output to float32")
    parser.add_argument("--presets", nargs="+", choices=PRESETS, default=list(PRESETS))
    parser.add_argument("--batches", nargs="+", type=int)
    parser.add_argument("--bank-size", type=int, default=32)
    args = parser.parse_args()
    if min(args.batches or [1]) < 1 or args.bank_size < 1 or args.threads < 1 or args.batch_size < 1:
        parser.error("Batch sizes, bank size and threads must be positive")
    if args.phase == "qualification" and args.count is None:
        args.count = 256
    run(args)


if __name__ == "__main__":
    main()
