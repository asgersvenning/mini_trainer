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


def test_rows(index: Path) -> dict:
    """Preserve labels and ordering from exactly the supplied test split."""
    data = json.loads(index.read_text())
    count = len(data["path"])
    if any(len(data[key]) != count for key in ("split", "label")):
        raise ValueError("Index path, split and label lengths differ")
    selected = [i for i, split in enumerate(data["split"]) if split == "test"]
    if not selected:
        raise ValueError("No test rows in saved index")
    result = {k: [v[i] for i in selected] for k, v in data.items() if isinstance(v, list) and len(v) == count}
    result["path"] = [str((index.parent / p).absolute()) for p in result["path"]]
    return result


def stage(source: Path, destination: Path, output: Path, limit: int, byte_limit: int, workers: int, data_index: Path | None = None) -> None:
    start = time.monotonic()
    print(
        f"Selecting all saved test rows from {data_index}"
        if data_index
        else f"Selecting up to {limit} JPEG/PNG files across class folders",
        flush=True,
    )
    index = test_rows(data_index) if data_index is not None else None
    paths = select_images(source, limit) if index is None else [Path(p) for p in index["path"]]
    print(f"Inspecting sizes of {len(paths)} selected files", flush=True)
    if not paths:
        raise RuntimeError("No JPEG/PNG candidates found")
    sizes = [path.stat().st_size for path in paths]
    total = sum(sizes)
    if total > byte_limit:
        raise RuntimeError(f"Selected files need {total} bytes; limit is {byte_limit}. Adjust the staging budget or subset size.")
    if shutil.disk_usage(destination.parent).free < total + 256 * 2**20:
        raise RuntimeError("Insufficient staging space with 256 MiB headroom")
    destination.mkdir(exist_ok=False)

    def relative_path(i: int, path: Path) -> Path:
        return path.relative_to(source) if index is None else Path(str(i // 4096)) / f"{i}{path.suffix}"

    def copy(item: tuple[int, Path]) -> dict:
        i, path = item
        relative = relative_path(i, path)
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
        remaining = iter(enumerate(paths))
        pending = set()
        exhausted = False
        while pending or not exhausted:
            while not exhausted and len(pending) < workers * 2:
                item = next(remaining, None)
                if item is None:
                    exhausted = True
                else:
                    pending.add(pool.submit(copy, item))
            if not pending:
                break
            done, pending = concurrent.futures.wait(pending, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                rows.append(future.result())
                if len(rows) % 4096 == 0 or len(rows) == len(paths):
                    print(f"Copied {len(rows)}/{len(paths)}; elapsed {time.monotonic() - start:.1f}s", flush=True)
    report = {"scope": "staging_trial_subset", "seconds": time.monotonic() - start, "bytes": total, "files": rows}
    if index is not None:
        index["path"] = [str(destination / relative_path(i, p)) for i, p in enumerate(paths)]
        (output / "staged-index.json").write_text(json.dumps(index) + "\n")
        report["scope"] = "supplied_test_split"
        report["source_index"] = str(data_index)
    (output / "staging.json").write_text(json.dumps(report, indent=2) + "\n")


def reuse_stage(previous: Path) -> tuple[Path, dict]:
    """Require a completed manifest and intact staged files, without source reads."""
    config = json.loads((previous / "inference.yaml").read_text())
    report = json.loads((previous / "staging.json").read_text())
    destination = Path(config["input"]).resolve(strict=True)
    rows = report["files"]
    if not rows:
        raise RuntimeError("Completed staging manifest contains no files")
    for row in rows:
        path = Path(row["staged"]).resolve(strict=True)
        if not path.is_relative_to(destination) or path.stat().st_size != row["bytes"]:
            raise RuntimeError(f"Staged file does not match manifest: {path}")
    return destination, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Fresh persistent trial directory")
    parser.add_argument("--images", type=int, default=1024)
    parser.add_argument("--data-index", type=Path, help="Stage all test rows from this saved index; ignores --images")
    parser.add_argument("--max-mib", type=int, default=2048)
    parser.add_argument("--copy-workers", type=int, default=4)
    parser.add_argument("--stage-timeout", type=int, default=300)
    parser.add_argument("--inference-timeout", type=int, default=600)
    parser.add_argument("--gpu", default="1")
    parser.add_argument("--reuse-stage", type=Path, help="Reuse a completed trial directory without copying source images")
    parser.add_argument("--stage-only", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.images, args.max_mib, args.copy_workers, args.stage_timeout, args.inference_timeout) <= 0:
        parser.error("Limits and worker counts must be positive")
    args.source = args.source.resolve(strict=True)
    args.weights = args.weights.resolve(strict=True)
    args.output = args.output.resolve()
    destination = Path("/dev/shm") / ("mt-expert-" + args.output.name)
    if args.stage_only:
        stage(args.source, destination, args.output, args.images, args.max_mib * 2**20, args.copy_workers, args.data_index)
        return

    from mini_trainer.data import auto_find_images

    if "cls2idx" not in inspect.signature(auto_find_images).parameters:
        raise RuntimeError("Install the focused inference discovery fix before running this trial")
    previous_report = None
    if args.reuse_stage is not None:
        destination, previous_report = reuse_stage(args.reuse_stage)
    elif destination.exists():
        raise RuntimeError(f"Staging directory already exists: {destination}; choose a fresh output name")
    args.output.mkdir(parents=True, exist_ok=False)
    if previous_report is not None:
        (args.output / "staging.json").write_text(json.dumps(previous_report, indent=2) + "\n")
        print(f"Reusing completed staging from {args.reuse_stage}: {destination}", flush=True)
    config = {"input": str(destination), "weights": str(args.weights)}
    if args.data_index is not None:
        if args.reuse_stage is not None:
            shutil.copyfile(args.reuse_stage / "staged-index.json", args.output / "staged-index.json")
        config["data_index"] = str(args.output / "staged-index.json")
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
    if args.reuse_stage is not None:
        commands = commands[1:]
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
    print("Check staging.json for the selected scope and exact file count.", flush=True)


if __name__ == "__main__":
    main()
