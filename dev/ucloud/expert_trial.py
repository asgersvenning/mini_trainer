"""Bounded RAM-staging trial followed by the normal hierarchical prediction CLI."""

import argparse
import concurrent.futures
import inspect
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

SUFFIXES = {".jpg", ".jpeg", ".png"}


def select_images(source: Path, limit: int) -> list[Path]:
    """Select round-robin across class folders; this is a trial, not a benchmark."""
    folders = sorted(path for path in source.iterdir() if path.is_dir() and not path.name.startswith("."))
    streams = [iter(path.rglob("*")) for path in folders]
    selected = []
    while streams and len(selected) < limit:
        active = []
        for stream in streams:
            path = next((p for p in stream if p.suffix.lower() in SUFFIXES and p.is_file()), None)
            if path is not None:
                selected.append(path)
                active.append(stream)
                if len(selected) == limit:
                    break
        streams = active
    return selected


def stage(source: Path, destination: Path, output: Path, limit: int, byte_limit: int, workers: int) -> None:
    start = time.monotonic()
    print(f"Selecting up to {limit} JPEG/PNG files across class folders", flush=True)
    paths = select_images(source, limit)
    if not paths:
        raise RuntimeError("No JPEG/PNG candidates found")
    sizes = [path.stat().st_size for path in paths]
    total = sum(sizes)
    if total > byte_limit:
        raise RuntimeError(f"Selected files need {total} bytes; limit is {byte_limit}. Reduce --images.")
    if shutil.disk_usage(destination.parent).free < total + 256 * 2**20:
        raise RuntimeError("Insufficient staging space with 256 MiB headroom")
    destination.mkdir(exist_ok=False)

    def copy(path: Path) -> dict:
        relative = path.relative_to(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        before = path.stat()
        shutil.copyfile(path, target)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or target.stat().st_size != before.st_size:
            raise RuntimeError(f"Source changed or copy was incomplete: {path}")
        return {"source": str(path), "staged": str(target), "bytes": before.st_size}

    rows = []
    print(f"Copying {len(paths)} files ({total / 2**20:.1f} MiB), {workers} readers", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(copy, path) for path in paths]
        for future in concurrent.futures.as_completed(futures):
            rows.append(future.result())
            if len(rows) % 32 == 0 or len(rows) == len(paths):
                print(f"Copied {len(rows)}/{len(paths)}; elapsed {time.monotonic() - start:.1f}s", flush=True)
    report = {"scope": "staging_trial_subset", "seconds": time.monotonic() - start, "bytes": total, "files": rows}
    (output / "staging.json").write_text(json.dumps(report, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Fresh persistent trial directory")
    parser.add_argument("--images", type=int, default=1024)
    parser.add_argument("--max-mib", type=int, default=2048)
    parser.add_argument("--copy-workers", type=int, default=4)
    parser.add_argument("--stage-timeout", type=int, default=300)
    parser.add_argument("--inference-timeout", type=int, default=600)
    parser.add_argument("--gpu", default="1")
    parser.add_argument("--stage-only", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.images, args.max_mib, args.copy_workers, args.stage_timeout, args.inference_timeout) <= 0:
        parser.error("Limits and worker counts must be positive")
    args.source = args.source.resolve(strict=True)
    args.weights = args.weights.resolve(strict=True)
    args.output = args.output.resolve()
    destination = Path("/dev/shm") / ("mt-expert-" + args.output.name)
    if args.stage_only:
        stage(args.source, destination, args.output, args.images, args.max_mib * 2**20, args.copy_workers)
        return

    from mini_trainer.data import auto_find_images

    if "cls2idx" not in inspect.signature(auto_find_images).parameters:
        raise RuntimeError("Install the focused inference discovery fix before running this trial")
    if destination.exists():
        raise RuntimeError(f"Staging directory already exists: {destination}; choose a fresh output name")
    args.output.mkdir(parents=True, exist_ok=False)
    config = {"input": str(destination), "weights": str(args.weights)}
    config_path = args.output / "inference.yaml"
    # JSON is valid YAML; no extra dependency or hand-edited config required.
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", MPLBACKEND="Agg")
    cli = Path(sys.executable).parent / "mt_hpredict"
    if not cli.is_file():
        raise RuntimeError(f"Missing installed CLI: {cli}")
    commands = [
        ([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--stage-only"], args.stage_timeout, "stage"),
        (
            [str(cli), "--config", str(config_path), "--output", str(args.output), "--name", "predictions"],
            args.inference_timeout,
            "inference",
        ),
    ]
    for command, timeout, phase in commands:
        print(f"Starting {phase}; timeout {timeout}s. Log: {args.output / (phase + '.log')}", flush=True)
        started = time.monotonic()
        with (args.output / f"{phase}.log").open("w") as log:
            with subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True) as proc:
                try:
                    returncode = proc.wait(timeout=timeout)
                except (subprocess.TimeoutExpired, KeyboardInterrupt):
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        pass
                    # Also stop loader descendants, even if the parent exited first.
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    raise SystemExit(f"{phase} interrupted or exceeded {timeout}s; see {log.name}.") from None
            if returncode:
                raise SystemExit(f"{phase} failed ({returncode}); see {log.name}")
        print(f"{phase} completed in {time.monotonic() - started:.1f}s", flush=True)
    print(f"Trial complete: {args.output}. Staged bytes retained at {destination}.", flush=True)
    print("This subset checks storage and inference functionality, not full expert benchmark accuracy.", flush=True)


if __name__ == "__main__":
    main()
