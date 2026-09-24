"""Read-only progress/ETA monitor for existing MAMBO collection logs; standard library only."""

import argparse
import json
import re
import sys
import time
from collections import deque
from pathlib import Path

PROGRESS = re.compile(r"^(?:(?:torch|onnx) \S+: )?(\d+)[ /](\d+)\s*$", re.MULTILINE)


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None  # The producer may be between truncating and writing its report.


def checkpoint(path):
    try:
        with path.open("rb") as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - 65536))
            matches = list(PROGRESS.finditer(stream.read().decode(errors="replace")))
        if matches:
            done, total = map(int, matches[-1].groups())
            if 0 < done <= total:
                return done, total, path.stat().st_mtime
    except FileNotFoundError:
        pass
    return None


class Rate:
    def __init__(self):
        self.points = deque()

    def estimate(self, done, total, timestamp, now):
        if self.points and (done < self.points[-1][1] or timestamp < self.points[-1][0]):
            self.points.clear()
        if not self.points or done > self.points[-1][1]:
            self.points.append((timestamp, done))
        while len(self.points) > 2 and self.points[1][0] < timestamp - 300:
            self.points.popleft()
        if len(self.points) < 2:
            return None
        first, last = self.points[0], self.points[-1]
        elapsed = last[0] - first[0]
        if elapsed <= 0:
            return None
        cadence = elapsed / (len(self.points) - 1)
        if now - last[0] > max(60, 3 * cadence):
            return None  # Do not continue projecting throughput while logs stop advancing.
        rate = (last[1] - first[1]) / elapsed
        return rate, (total - done) / rate


def duration(seconds):
    seconds = max(0, int(seconds))
    return f"{seconds // 3600}h {(seconds % 3600) // 60:02}m {seconds % 60:02}s"


def snapshot(directory, rates, now):
    plan = read_json(directory / "plan.json")
    if plan is None:
        return [f"Waiting for readable {directory / 'plan.json'}"], False
    rows = []
    completed = 0
    for job in plan["jobs"]:
        name = job["name"]
        report = read_json(directory / name / "report.json")
        status = report.get("status") if report else None
        if name in plan["completed"] or status == "complete":
            rows.append(f"{name}: complete")
            completed += 1
            continue
        log = directory / f"{name}.log"
        point = checkpoint(log)
        if point is None:
            state = status or ("initializing; no image checkpoint yet" if log.exists() else "queued")
            rows.append(f"{name}: {state}")
            continue
        done, total, timestamp = point
        estimate = rates.setdefault(name, Rate()).estimate(done, total, timestamp, now)
        text = f"{name}: {done:,}/{total:,} ({100 * done / total:.1f}%) | checkpoint age {duration(now - timestamp)}"
        if status == "failed":
            text += " | FAILED"
        elif done == total:
            text += " | finalizing; completion not yet confirmed"
        elif estimate:
            rate, eta = estimate
            text += f" | ~{rate:.1f} images/s | remaining at last checkpoint ~{duration(eta)}"
        else:
            text += " | ETA unavailable: need advancing checkpoints"
        rows.append(text)
    header = f"{directory.name}: {plan['status']} | {completed}/{len(plan['jobs'])} jobs complete"
    return [header, *rows], plan["status"] in ("complete", "failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Campaign phase directory, e.g. runs-ptx/full")
    parser.add_argument("--interval", type=float, default=10, help="Poll seconds (default 10)")
    parser.add_argument("--once", action="store_true", help="Print counts/status once; rate needs multiple observations")
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval must be positive")
    rates = {}
    try:
        while True:
            rows, finished = snapshot(args.directory.expanduser(), rates, time.time())
            if sys.stdout.isatty() and not args.once:
                print("\033[2J\033[H", end="")
            print(time.strftime("%Y-%m-%d %H:%M:%S"), *rows, sep="\n", flush=True)
            if args.once or finished:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
