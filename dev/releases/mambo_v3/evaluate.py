"""Stream release predictions once per model variant and reduce all declared presets."""

import argparse
import csv
import hashlib
import platform
import time
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from deployment.mambo_deploy import Predictor
from deployment.mambo_deploy.augmentation import DEFAULT_TTA, PROFILES
from deployment.mambo_deploy.preprocessing import prepare_batch as prepare_batch
from deployment.mambo_deploy.result_worker import ResultWorker
from deployment.mambo_deploy.results import Prediction
from dev.benchmarks.inference.onnx_inference import file_hash
from dev.releases.mambo_v3.evaluation_data import CSV_COLUMNS, PRESETS, canonical_rows, load_records, prepare_flemming, write_json


def runtime_settings(threads, backend="torch"):
    result = {"threads": threads, "tf32": False, "autocast": False, "precision": "float32", "platform": platform.platform()}
    if backend == "torch":
        import torch

        torch.set_num_threads(threads)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        result.update(torch=torch.__version__, cuda=torch.version.cuda)
    else:
        import onnxruntime as ort

        result["onnxruntime"] = ort.__version__
    return result


def collect(args):
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "running",
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "runner_sha256": file_hash(__file__),
        "manifest_sha256": file_hash(args.manifest),
        "bundle_sha256": file_hash(args.bundle / "release.json"),
        "runtime": runtime_settings(args.threads, args.backend),
    }
    start = time.perf_counter()
    try:
        manifest, records = load_records(args.manifest, args.root, args.count, args.seed)
        report.update(samples=len(records), species=len({r["labels"][0] for r in records}), dataset=manifest["dataset"])
        write_json(output / "samples.json", records)
        report["sample_ids_sha256"] = file_hash(output / "samples.json")
        predictor = Predictor(
            args.bundle,
            backend=args.backend,
            device=args.device,
            model="full",
            threads=args.threads,
            precision=args.precision,
            tta=getattr(args, "tta", "none"),
        )
        report["effective_precision"] = predictor.effective_precision
        report["runtime"].update(
            precision=predictor.effective_precision,
            autocast=predictor.effective_precision in ("fp16", "bf16"),
            tf32=predictor.effective_precision == "tf32",
        )
        selectors = {name: Predictor(args.bundle, model=name).selected for name in args.presets}
        # A preset supplied as a custom list must produce the same mask/order.
        custom = Predictor(args.bundle, class_list=predictor.bundle.file(predictor.bundle.regions["europe_v3"]["path"]))
        np.testing.assert_array_equal(custom.selected, Predictor(args.bundle, model="europe_v3").selected)
        report["preset_as_custom_equivalent"] = True
        report["lists"] = {
            name: {
                "count": len(selected),
                "sha256": hashlib.sha256(
                    ("\n".join(predictor.bundle.classes["labels"][0][i] for i in selected) + "\n").encode()
                ).hexdigest(),
            }
            for name, selected in selectors.items()
        }
        with ExitStack() as stack:
            writers = {}
            for name in selectors:
                directory = output / name
                directory.mkdir()
                stream = stack.enter_context((directory / "mini_metric.csv").open("w", newline=""))
                writers[name] = csv.writer(stream)
                writers[name].writerow(CSV_COLUMNS)
            embeddings = None
            if args.embeddings:
                embeddings = np.lib.format.open_memmap(output / "embeddings.npy", mode="w+", dtype=np.float32, shape=(len(records), 1280))
            timings = {
                name: 0.0
                for name in (
                    "input_wait_seconds",
                    "runtime_submit_seconds",
                    "output_wait_seconds",
                    "hierarchy_seconds",
                    "prediction_seconds",
                    "write_seconds",
                    "model_stream_seconds",
                    "d2h_device_seconds",
                    "output_completion_wait_seconds",
                )
            }

            plans = {name: predictor.hierarchy_plan(selected) for name, selected in selectors.items()}

            def process(batch, resolve, offset):
                leaf, vectors, ranks = resolve()
                if leaf.shape != (len(batch), len(predictor.bundle.classes["labels"][0])) or not np.isfinite(leaf).all():
                    raise ValueError("Invalid leaf scores")
                if embeddings is not None:
                    if vectors.shape != (len(batch), 1280) or not np.isfinite(vectors).all():
                        raise ValueError("Invalid embeddings")
                    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4)
                    embeddings[offset : offset + len(batch)] = vectors
                phase = {name: 0.0 for name in ("hierarchy_seconds", "prediction_seconds", "write_seconds")}
                for name, selected in selectors.items():
                    t = time.perf_counter()
                    reduced = ranks[name] if ranks is not None else plans[name].numpy(leaf)
                    phase["hierarchy_seconds"] += time.perf_counter() - t
                    t = time.perf_counter()
                    result = Prediction(*reduced)
                    phase["prediction_seconds"] += time.perf_counter() - t
                    t = time.perf_counter()
                    writers[name].writerows(canonical_rows(batch, result, offset))
                    phase["write_seconds"] += time.perf_counter() - t
                return len(batch), phase

            worker = stack.enter_context(ResultWorker(process))

            def finish_one():
                t = time.perf_counter()
                count, phase = worker.pop()
                timings["output_wait_seconds"] += time.perf_counter() - t
                for name, value in phase.items():
                    timings[name] += value
                return count

            report["pipeline"] = {"decode_workers": args.decode_workers, "prefetch_batches": args.prefetch_batches, "read_once": True}
            stream_stats = {}
            report["streaming"] = stream_stats
            batches = predictor.prepared_batches(
                ((args.root / record["path"], record["sha256"]) for record in records),
                args.batch_size,
                read_workers=args.read_workers,
                prepare_workers=max(1, args.decode_workers),
                read_window=args.read_window,
                prefetch_batches=args.prefetch_batches,
                encoded_budget=args.encoded_budget_mib * 1024**2,
                stats=stream_stats,
                device_prefetch=not args.no_device_prefetch,
            )
            stack.callback(batches.close)
            completed = 0
            last_progress = time.perf_counter()
            previous_completed = 0
            previous_timings = dict(timings)
            previous_preparation = 0.0
            while True:
                waiting = time.perf_counter()
                try:
                    offset, views, batch_count = next(batches)
                    batch = records[offset : offset + batch_count]
                except StopIteration:
                    while worker.pending:
                        completed += finish_one()
                    print(f"{args.backend} {args.device}: {completed}/{len(records)}", flush=True)
                    break
                timings["input_wait_seconds"] += time.perf_counter() - waiting
                t = time.perf_counter()
                resolve = predictor._ranked_views(views, len(views), selectors, args.embeddings, defer=True)
                timings["runtime_submit_seconds"] += time.perf_counter() - t
                timings.update(predictor.runtime_timings)
                worker.submit(batch, resolve, offset)

                if len(worker.pending) == 2:
                    completed += finish_one()
                now = time.perf_counter()
                if now - last_progress >= 5 or completed == len(records):
                    interval = now - last_progress
                    rate = (completed - previous_completed) / interval
                    phase_seconds = {name: round(value - previous_timings[name], 3) for name, value in timings.items()}
                    preparation = stream_stats.get("preparation_worker_seconds", 0.0)
                    phase_seconds["background_preparation_worker_seconds"] = round(preparation - previous_preparation, 3)
                    # Retain the existing machine-readable count line for live monitors.
                    print(f"{args.backend} {args.device}: {completed}/{len(records)}", flush=True)
                    eta = f"{(len(records) - completed) / rate / 60:.1f} min" if rate else "waiting for first written batch"
                    print(f"{rate:.1f} images/s; ETA {eta}; pipeline={stream_stats}", flush=True)
                    print(f"interval={interval:.3f}s; phases={phase_seconds}", flush=True)
                    previous_timings = dict(timings)
                    previous_completed = completed
                    previous_preparation = preparation
                    last_progress = now
            timings.update(predictor.runtime_timings)
            if embeddings is not None:
                embeddings.flush()
        if args.backend == "onnx":
            import onnxruntime as ort

            report["runtime"]["onnxruntime"] = ort.__version__
            report["onnx_session_info"] = predictor.onnx_session_info
            report["providers"] = {key: session.get_providers() for key, session in predictor._sessions.items()}
            report["provider_options"] = {key: session.get_provider_options() for key, session in predictor._sessions.items()}
        report.update(status="complete", timings=timings)
        report["csv_sha256"] = {name: file_hash(output / name / "mini_metric.csv") for name in selectors}
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - start
        write_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--root", type=Path, required=True)
    prep.add_argument("--reference", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    run = commands.add_parser("collect")
    for name in ("bundle", "manifest", "root", "output"):
        run.add_argument(f"--{name}", type=Path, required=True)
    run.add_argument("--backend", choices=["torch", "onnx"], required=True)
    run.add_argument("--device", default="cpu")
    run.add_argument("--precision", choices=["auto", "fp32", "fp16", "bf16", "tf32"], default="fp32")
    run.add_argument("--tta", nargs="?", const=DEFAULT_TTA, choices=PROFILES, default="none")
    run.add_argument("--embeddings", action="store_true")
    run.add_argument("--count", type=int)
    run.add_argument("--seed", type=int, default=20260923)
    run.add_argument("--batch-size", type=int, default=32)
    run.add_argument("--threads", type=int, default=4)
    run.add_argument("--decode-workers", type=int, default=4)
    run.add_argument("--prefetch-batches", type=int, default=2, help="Prepared batches queued ahead; 0 disables overlap")
    run.add_argument("--no-device-prefetch", action="store_true")
    run.add_argument("--read-workers", type=int, default=32)
    run.add_argument("--read-window", type=int, default=128)
    run.add_argument("--encoded-budget-mib", type=int, default=256)
    run.add_argument("--presets", nargs="+", default=list(PRESETS))
    args = parser.parse_args()
    if args.command == "prepare":
        prepare_flemming(args.root, args.reference, args.output)
    else:
        if args.batch_size < 1 or args.decode_workers < 0 or args.prefetch_batches < 0:
            parser.error("batch-size must be positive; decode-workers and prefetch-batches must be nonnegative")
        collect(args)


if __name__ == "__main__":
    main()
