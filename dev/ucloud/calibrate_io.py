#!/usr/bin/env python3
"""Standalone filesystem image-I/O concurrency calibration (Python 3.12+, POSIX).

Examples:
  python calibrate_io.py /images --mode stage --destination /dev/shm --output /work/io.json
  python calibrate_io.py /images --mode decode --resize 384 --output /work/decode.json

Read an existing report without its path manifest:
  python calibrate_io.py --summary /work/io.json

Only decode mode requires Pillow. No mini_trainer, PyTorch, root access or cache
flushing. All trials use disjoint selections; externally warmed caches are unknown.
"""

import argparse
import json
import os
import random
import resource
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def write_json(path, value):
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def discover(source, limit, deadline, rng):
    """Shuffle traversal before bounded discovery; never read source contents."""
    paths = []
    for directory, folders, files in os.walk(source):
        rng.shuffle(folders)
        rng.shuffle(files)
        for name in files:
            if Path(name).suffix.lower() in EXTENSIONS:
                paths.append(str(Path(directory) / name))
                if len(paths) >= limit:
                    rng.shuffle(paths)
                    return paths, True
        if time.monotonic() >= deadline:
            raise TimeoutError("Budget exhausted enumerating paths")
    rng.shuffle(paths)
    return paths, False


def warm_pool(pool, workers):
    """Create every thread before starting any filesystem work."""
    barrier = threading.Barrier(workers + 1)
    try:
        futures = [pool.submit(barrier.wait) for _ in range(workers)]
    except BaseException:
        barrier.abort()
        raise
    barrier.wait()
    for future in futures:
        future.result()


def child(job_path):
    initialized = time.monotonic()
    job = json.loads(Path(job_path).read_text())
    image_module = None
    if job["mode"] == "decode":
        from PIL import Image

        image_module = Image
    started = time.monotonic()

    def operation(item):
        index, source = item
        began = time.monotonic()
        if image_module is not None:
            with image_module.open(source) as image:
                converted = image.convert("RGB")
                if job["resize"]:
                    resized = converted.resize((job["resize"], job["resize"]), image_module.Resampling.BICUBIC)
                    resized.close()
                converted.close()
            size = os.stat(source).st_size
        else:
            size = 0
            target = open(Path(job["destination"]) / str(index), "xb") if job["mode"] == "stage" else None
            try:
                with open(source, "rb") as stream:
                    while chunk := stream.read(1024 * 1024):
                        if target is not None:
                            target.write(chunk)
                        size += len(chunk)
            finally:
                if target is not None:
                    target.close()
        return size, time.monotonic() - began

    rows, errors = [], []
    first_completion = None

    def snapshot(finished=False):
        elapsed = time.monotonic() - started
        write_json(
            job["result"],
            {
                "completed": len(rows),
                "bytes": sum(row[0] for row in rows),
                "seconds": elapsed,
                "images_per_second": len(rows) / elapsed,
                "mib_per_second": sum(row[0] for row in rows) / 2**20 / elapsed,
                "latency_median_seconds": statistics.median(row[1] for row in rows) if rows else None,
                "latency_p95_seconds": sorted(row[1] for row in rows)[int((len(rows) - 1) * 0.95)] if rows else None,
                "first_completion_seconds": first_completion,
                "errors": errors,
                "finished": finished,
                "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            },
        )

    items = iter(enumerate(job["paths"]))
    # Bound outstanding tasks and buffers instead of queuing the whole dataset.
    with open(job["attempted"], "wb", buffering=0) as journal, ThreadPoolExecutor(max_workers=job["workers"]) as pool:
        warm_pool(pool, job["workers"])
        started = time.monotonic()
        write_json(job["ready"], {"started": started, "initialization_seconds": started - initialized})
        next_snapshot = started + 0.25

        def submit(item):
            # Record BEFORE submission. After a timeout only this prefix may
            # have read source bytes; the untouched suffix remains available.
            journal.write(b"x")
            return pool.submit(operation, item)

        pending = {submit(item) for item in list_next(items, job["workers"])}
        while pending:
            done, pending = wait(pending, timeout=0.25, return_when=FIRST_COMPLETED)
            for future in done:
                try:
                    rows.append(future.result())
                    if first_completion is None:
                        first_completion = time.monotonic() - started
                except Exception as error:
                    errors.append(f"{type(error).__name__}: {error}")
                item = next(items, None)
                if item is not None and not errors:
                    pending.add(submit(item))
            if errors or time.monotonic() >= next_snapshot:
                snapshot()
                next_snapshot = time.monotonic() + 0.25
            if errors:
                break
    snapshot(finished=True)


def list_next(iterator, count):
    result = []
    for _ in range(count):
        value = next(iterator, None)
        if value is None:
            break
        result.append(value)
    return result


def stop(proc):
    """Do not launch another trial while an uninterruptible reader survives."""
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"I/O process {proc.pid} remains blocked after termination; stop calibration here") from None


def run_trial(args, paths, workers, phase, work, deadline):
    started = time.monotonic()
    byte_limit = args.max_mib * 2**20
    if args.mode == "stage":
        byte_limit = min(byte_limit, max(0, shutil.disk_usage(args.destination).free - 256 * 2**20))
    sizes = []
    total = 0
    for path in paths:
        size = os.stat(path).st_size
        if total + size > byte_limit:
            break
        sizes.append(size)
        total += size
    if not sizes:
        raise RuntimeError(f"A single image exceeds available trial space ({byte_limit / 2**20:.1f} MiB): {paths[0]}")
    paths = paths[: len(sizes)]
    requested_workers = workers
    workers = min(workers, len(paths))
    result_path, job_path = work / "result.json", work / "job.json"
    attempted_path = work / "attempted.bin"
    ready_path = work / "ready.json"
    ready_path.unlink(missing_ok=True)
    result_path.unlink(missing_ok=True)
    attempted_path.unlink(missing_ok=True)
    destination = None
    if args.mode == "stage":
        if shutil.disk_usage(args.destination).free < sum(sizes) + 256 * 2**20:
            raise RuntimeError("Staging destination has insufficient free space")
        destination = tempfile.mkdtemp(prefix="mt-calibrate-", dir=args.destination)
    job = {
        "paths": paths,
        "workers": workers,
        "mode": args.mode,
        "resize": args.resize,
        "destination": destination,
        "result": str(result_path),
        "attempted": str(attempted_path),
        "ready": str(ready_path),
    }
    write_json(job_path, job)
    proc = None
    reason = None
    try:
        with (work / "child.log").open("w") as log:
            proc = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), "--child", str(job_path)], stdout=log, stderr=log, start_new_session=True
            )
            until = min(deadline, time.monotonic() + getattr(args, "startup_seconds", 30))
            ready = None
            while proc.poll() is None:
                if ready is None and ready_path.exists():
                    ready = json.loads(ready_path.read_text())
                    until = min(deadline, ready["started"] + args.trial_seconds)
                if time.monotonic() >= until:
                    reason = "time_limit" if ready is not None else "startup_time_limit"
                    stop(proc)
                    break
                # Linux exposes current RSS; final peak RSS is recorded by the child.
                status = Path(f"/proc/{proc.pid}/status")
                if status.exists():
                    try:
                        rss = next((int(line.split()[1]) for line in status.read_text().splitlines() if line.startswith("VmRSS:")), 0)
                    except FileNotFoundError:
                        rss = 0
                    if rss > args.max_rss_mib * 1024:
                        reason = "memory_limit"
                        stop(proc)
                        break
                time.sleep(0.1)
            if ready is None and ready_path.exists():
                ready = json.loads(ready_path.read_text())
            row = (
                json.loads(result_path.read_text())
                if result_path.exists()
                else {
                    "completed": 0,
                    "images_per_second": 0,
                    "seconds": time.monotonic() - ready["started"] if ready is not None else 0,
                    "errors": [],
                    "finished": False,
                }
            )
            row.update(
                workers=workers,
                requested_workers=requested_workers,
                initialization_seconds=ready["initialization_seconds"] if ready is not None else None,
                attempted=attempted_path.stat().st_size if attempted_path.exists() else 0,
                phase=phase,
                selected=len(paths),
                selected_bytes=sum(sizes),
                termination=reason,
                wall_seconds=time.monotonic() - started,
            )
            if proc.returncode and reason is None:
                row["errors"].append((work / "child.log").read_text()[-2000:])
            row["eligible"] = not row["errors"] and reason != "memory_limit" and row["completed"] > 0
            return row
    finally:
        if proc is not None and proc.poll() is None:
            stop(proc)
        if destination is not None:
            shutil.rmtree(destination)


def recommend(rows, tolerance):
    confirmations = [row for row in rows if row["phase"] == "confirm"]
    considered = confirmations if confirmations else rows
    scores = {}
    for workers in sorted({row["workers"] for row in considered}):
        group = [row for row in considered if row["workers"] == workers]
        if group and all(row["eligible"] for row in group):
            scores[workers] = statistics.median(row["images_per_second"] for row in group)
    if not scores:
        if confirmations:
            failed = {row["workers"] for row in confirmations if row.get("errors") or row.get("termination") == "memory_limit"}
            provisional = recommend([row for row in rows if row["phase"] == "sweep" and row["workers"] not in failed], tolerance)
            provisional["confirmation_note"] = "No usable confirmation; showing sweep estimate only"
            return provisional
        return {"workers": None, "reason": "No eligible trials"}
    best = max(scores.values())
    selected = min(workers for workers, rate in scores.items() if rate >= best * (1 - tolerance))
    return {
        "workers": selected,
        "median_images_per_second": scores[selected],
        "scores": scores,
        "basis": "confirmation" if confirmations else "provisional_sweep",
        "near_best_tolerance": tolerance,
    }


def summary(report):
    lines = [f"Mode: {report.get('mode', 'unknown')} | Source: {report.get('source', 'unknown')}"]
    for row in report.get("trials", []):
        lines.append(
            f"{row['phase']:7} {row['workers']:4} threads | {row['images_per_second']:.1f} images/s | "
            f"{row['completed']} complete | {row.get('termination') or 'finished'}"
        )
        if row.get("errors"):
            lines.append("  error: " + str(row["errors"][-1]).strip().split("\n")[-1][:200])
    lines.append("Recommendation: " + json.dumps(report.get("recommendation")))
    if report.get("error"):
        lines.append("Failure: " + report["error"])
    return "\n".join(lines) + "\n"


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--summary":
        print(summary(json.loads(Path(sys.argv[2]).read_text())), end="")
        return
    if len(sys.argv) == 3 and sys.argv[1] == "--child":
        child(sys.argv[2])
        return
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path)
    parser.add_argument("--mode", choices=("read", "stage", "decode"), default="read")
    parser.add_argument("--destination", type=Path, help="Required for stage; use the intended real staging filesystem")
    parser.add_argument("--output", type=Path, default=Path("io-calibration.json"))
    parser.add_argument("--workers", default="1,4,16,64,128,256,512,1024")
    parser.add_argument("--files-per-trial", type=int, default=2048, help="Maximum per trial; only submitted paths are consumed")
    parser.add_argument("--trial-seconds", type=float, default=30, help="I/O measurement time, excluding initialization")
    parser.add_argument("--startup-seconds", type=float, default=30, help="Separate process/thread initialization limit")
    parser.add_argument("--budget-seconds", type=float, default=600)
    parser.add_argument("--confirmation-rounds", type=int, default=2)
    parser.add_argument("--max-mib", type=int, default=16384, help="Encoded byte cap per trial; sample size shrinks automatically")
    parser.add_argument("--max-rss-mib", type=int, default=4096, help="Child RSS stop threshold on Linux")
    parser.add_argument("--resize", type=int, default=0, help="Optional square bicubic resize in decode mode")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tolerance", type=float, default=0.05, help="Prefer lowest concurrency within this fraction of best")
    args = parser.parse_args()
    workers = sorted(set(int(value) for value in args.workers.split(",")))
    if (
        not workers
        or min(workers) < 1
        or min(args.files_per_trial, args.trial_seconds, args.startup_seconds, args.budget_seconds, args.max_mib, args.max_rss_mib) <= 0
    ):
        parser.error("Concurrency and limits must be positive")
    if args.confirmation_rounds < 0 or args.resize < 0 or not 0 <= args.tolerance < 1:
        parser.error("Invalid confirmation rounds, resize or tolerance")
    args.source = args.source.resolve(strict=True)
    if args.mode == "stage":
        if args.destination is None:
            parser.error("--destination is required for stage mode")
        args.destination = args.destination.resolve(strict=True)
        if args.destination.is_relative_to(args.source):
            parser.error("Staging destination must be outside the source tree")
    original_output = args.output
    suffix = 2
    while args.output.exists():
        args.output = original_output.with_name(f"{original_output.stem}-{suffix}{original_output.suffix}")
        suffix += 1
    print(f"Report: {args.output}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if max(workers) > args.files_per_trial:
        workers = sorted({min(value, args.files_per_trial) for value in workers})
        print(f"Concurrency capped by --files-per-trial: {workers}", flush=True)
    rng = random.Random(args.seed)
    deadline = time.monotonic() + args.budget_seconds
    maximum = args.files_per_trial * (len(workers) + 3 * args.confirmation_rounds)
    print(f"Discovering up to {maximum} paths without reading image contents...", flush=True)
    paths, limited = discover(args.source, maximum, deadline, rng)
    report = {
        "mode": args.mode,
        "source": str(args.source),
        "destination": str(args.destination),
        "seed": args.seed,
        "cache_state": "unknown; disjoint selections, no cache flushing",
        "discovery_limited": limited,
        "discovered": len(paths),
        "trials": [],
        "selection_paths": [],
        "recommendation": None,
    }
    print(f"Found {len(paths)} candidates. Each path is assigned to at most one trial.", flush=True)
    cursor = 0

    def measure(concurrency, phase, work):
        nonlocal cursor
        count = min(args.files_per_trial, len(paths) - cursor)
        if count == 0 or time.monotonic() >= deadline:
            return False
        selected = paths[cursor : cursor + count]
        try:
            row = run_trial(args, selected, concurrency, phase, work, deadline)
        except BaseException:
            journal = work / "attempted.bin"
            attempted = journal.stat().st_size if journal.exists() else 0
            report["failed_trial"] = {"workers": concurrency, "phase": phase, "attempted": attempted}
            report["selection_paths"].append(selected[:attempted])
            cursor += attempted
            raise
        cursor += row["attempted"]
        report["trials"].append(row)
        report["selection_paths"].append(selected[: row["attempted"]])
        report["recommendation"] = recommend(report["trials"], args.tolerance)
        write_json(args.output, report)
        print(
            f"{phase:7} {row['workers']:4} threads (requested {concurrency}): {row['images_per_second']:9.1f} images/s; "
            f"{row['completed']} completed, {row['attempted']} attempted; {row['termination'] or 'finished'}",
            flush=True,
        )
        return True

    try:
        with tempfile.TemporaryDirectory(prefix="mt-io-calibration-") as temporary:
            work = Path(temporary)
            for concurrency in workers:
                if not measure(concurrency, "sweep", work):
                    break
            top = sorted((row for row in report["trials"] if row["eligible"]), key=lambda row: row["images_per_second"], reverse=True)[:3]
            finalists = list(dict.fromkeys(row["workers"] for row in top))
            for _ in range(args.confirmation_rounds):
                rng.shuffle(finalists)
                for concurrency in finalists:
                    if not measure(concurrency, "confirm", work):
                        break
    except (Exception, KeyboardInterrupt) as error:
        report["error"] = f"{type(error).__name__}: {error}"
        error_path = args.output.with_suffix(".error.log")
        error_path.write_text(traceback.format_exc())
        raise SystemExit(f"Calibration stopped: {error}. Details saved to {error_path}") from None
    finally:
        report["recommendation"] = recommend(report["trials"], args.tolerance)
        report["used_paths"] = cursor
        report["notes"] = [
            "Best tested thread concurrency, not a global optimum or a DataLoader process-worker setting.",
            "Decode measures CPU image open/RGB conversion and optional resize; no transforms, batching, GPU or DDP.",
            "Stage measures buffered writes and close, not fsync durability; read mode discards bytes.",
            "Process/thread initialization is excluded from I/O timing and recorded separately. Partial completed reads are eligible.",
            "Disjoint paths do not guarantee cold data, balanced file sizes, or independence from shared-service load.",
        ]
        write_json(args.output, report)
        summary_path = args.output.with_suffix(".summary.txt")
        summary_path.write_text(summary(report))
        print(f"Compact summary: {summary_path}", flush=True)
    if report["recommendation"]["workers"] is None:
        print("No completed reads to rank. Increase --trial-seconds or inspect the recorded errors.", flush=True)
    print(json.dumps(report["recommendation"], indent=2), flush=True)
    print(f"Full timings, errors and disjoint sample manifest: {args.output}", flush=True)


if __name__ == "__main__":
    main()
