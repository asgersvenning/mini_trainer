"""Launch the prototype explorer with a checkpoint path or browser file picker."""

import argparse
import hashlib
import importlib.util
import json
import os
import secrets
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import webbrowser
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit


def require_dependencies(angular=True):
    missing = [name for name in (["scipy", "sklearn"] if angular else ["scipy"]) if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError("Prototype exploration requires: " + ", ".join(missing) + ". Install mini_trainer[explorer].")


def generate(weights, output, *, angular=True, synthetic=False, threads=4, cache_dir=None):
    require_dependencies(angular)
    from .cache import analysis_key, read_analysis, store_analysis

    output = Path(output)
    key = contract = cached = None
    if cache_dir is not None:
        print("Checking analysis cache", flush=True)
        key, contract = analysis_key(weights, angular=angular, synthetic=synthetic, threads=threads)
        cached = read_analysis(cache_dir, key, contract)
    if cached is not None:
        from .explore import render_report

        output.mkdir(parents=True, exist_ok=True)
        for name, raw in cached.items():
            (output / name).write_bytes(raw)
        render_report(output, cached["report-data.json"].decode())
        print("Analysis cache hit; rendered current viewer", flush=True)
        return output / "explorer.html"
    import torch

    from .explore import create_report

    torch.set_num_threads(threads)
    result = create_report(weights, output, include_tsne=angular, synthetic=synthetic)
    if cache_dir is not None:
        store_analysis(cache_dir, key, contract, output)
        print("Analysis cache miss; stored completed result", flush=True)
    return result


class Workspace:
    """Owned preparation jobs; only the current job can publish a report."""

    def __init__(self, root, angular=True, synthetic=False, threads=4, cache_dir=None):
        self.root = root
        empty = root / "empty"
        empty.mkdir()
        self.public = root / "report"
        self.public.symlink_to(empty, target_is_directory=True)
        self.angular, self.synthetic, self.threads = angular, synthetic, threads
        self.cache_dir = cache_dir
        self.token = secrets.token_urlsafe(32)
        self.lock = threading.Lock()  # One upload at a time, independent of analysis.
        self.guard = threading.RLock()
        self.process = None
        self.job = 0
        self.workers = []
        self.state = {"state": "idle"}
        self.service = None
        self.ready = False

    def view_state_path(self, key):
        if not key or len(key) > 512:
            raise ValueError("Invalid view-state identity")
        return (self.cache_dir or self.root) / ".view-state" / (hashlib.sha256(key.encode()).hexdigest() + ".json")

    def read_view_state(self, key):
        path = self.view_state_path(key)
        try:
            return json.loads(path.read_text())
        except (OSError, ValueError):
            return None

    def write_view_state(self, key, value):
        path = self.view_state_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix("." + secrets.token_hex(8))
        try:
            temporary.write_text(json.dumps(value))
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def cancel(self):
        with self.guard:
            self.job += 1
            process = self.process
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            self.process = None
            self.state = {"state": "cancelled", "ready": self.ready}

    def close(self):
        self.cancel()
        for worker in self.workers:
            worker.join(timeout=10)

    def status(self):
        with self.guard:
            state = dict(self.state, ready=self.ready)
            if state["state"] == "running":
                state["elapsed"] = round(time.monotonic() - state["started"], 1)
                log = self.root / f"job-{self.job}" / "generation.log"
                if log.exists():
                    lines = log.read_text(errors="replace").splitlines()
                    stages = [line for line in lines if line and not line.startswith("{")]
                    state["stage"] = stages[-1] if stages else "Starting analysis"
            return state

    def start(self, weights):
        self.cancel()
        with self.guard:
            job = self.job
            self.state = {"state": "running", "file": weights.name, "started": time.monotonic(), "job": job}
            worker = threading.Thread(target=self.build, args=(weights, job), daemon=True)
            self.workers = [item for item in self.workers if item.is_alive()]
            self.workers.append(worker)
            worker.start()

    def build(self, weights, job):
        directory = self.root / f"job-{job}"
        directory.mkdir()
        output = directory / "report"
        process = None
        try:
            command = [
                sys.executable,
                "-m",
                "mini_trainer.visualization.prototype_space",
                str(weights),
                "--export",
                "--output",
                str(output),
                "--threads",
                str(self.threads),
            ]
            command += ["--cache-dir", str(self.cache_dir)] if self.cache_dir is not None else ["--no-cache"]
            if not self.angular:
                command.append("--pca-only")
            if self.synthetic:
                command.append("--synthetic")
            with (directory / "generation.log").open("w") as log:
                with self.guard:
                    if job != self.job:
                        return
                    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                    self.process = process
                result = process.wait()
            with self.guard:
                if job != self.job:
                    return
                if result:
                    raise RuntimeError((directory / "generation.log").read_text()[-4000:])
                data = json.loads((output / "report-data.json").read_text())
                ids = {name for case in data.values() if not case["metadata"].get("synthetic") for name in case["names"] if name.isdigit()}
                link = directory / "publish"
                link.symlink_to(output, target_is_directory=True)
                previous = self.public.resolve().parent
                os.replace(link, self.public)
                if previous.parent == self.root and previous.name.startswith("job-"):
                    shutil.rmtree(previous, ignore_errors=True)
                self.service.allowed_ids = ids
                self.ready = True
                log_text = (directory / "generation.log").read_text()
                cache_status = "hit" if "Analysis cache hit" in log_text else "miss" if self.cache_dir is not None else "disabled"
                self.state = {"state": "ready", "file": weights.name, "url": "/explorer.html", "cache": cache_status}
        except Exception as error:
            with self.guard:
                if job == self.job:
                    self.state = {"state": "error", "error": str(error)}
        finally:
            with self.guard:
                if self.public.resolve() != output.resolve():
                    shutil.rmtree(directory, ignore_errors=True)
                if weights.parent.parent == self.root / "inputs":
                    shutil.rmtree(weights.parent, ignore_errors=True)
                pid = getattr(process, "pid", None)
                if pid is not None and self.cache_dir is not None and self.cache_dir.exists():
                    for pending in self.cache_dir.glob(f".pending-{pid}-*"):
                        shutil.rmtree(pending, ignore_errors=True)


def session_handler(workspace):
    from .serve import Handler

    class SessionHandler(Handler):
        def do_GET(self):
            path = urlsplit(self.path).path
            if path == "/":
                page = Path(__file__).with_name("launcher.html").read_text().replace("__SESSION_TOKEN__", workspace.token).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(page)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(page)
                return
            if path == "/api/health":
                self.send_json({"photo_api": True, "model_picker": True})
                return
            if path == "/api/view-state":
                key = parse_qs(urlsplit(self.path).query).get("key", [""])[0]
                if not key or len(key) > 512:
                    self.send_json({"error": "Invalid view-state identity"}, 400)
                else:
                    self.send_json({"state": workspace.read_view_state(key), "token": workspace.token})
                return
            if path == "/api/session":
                self.send_json(workspace.status())
                return
            return super().do_GET()

        def do_POST(self):
            if (
                self.path not in {"/api/weights", "/api/cancel", "/api/view-state"}
                or self.headers.get("X-Explorer-Token") != workspace.token
            ):
                self.send_json({"error": "Invalid local session request"}, 403)
                return
            if self.path == "/api/view-state":
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if not 0 < length <= 65536:
                        raise ValueError("View state must be at most 64 KiB")
                    value = json.loads(self.rfile.read(length))
                    workspace.write_view_state(value["key"], value["state"])
                    self.send_json({"saved": True})
                except (OSError, ValueError, KeyError, TypeError) as error:
                    self.send_json({"error": str(error)}, 400)
                return
            if self.path == "/api/cancel":
                workspace.cancel()
                self.send_json(workspace.status())
                return
            if not workspace.lock.acquire(blocking=False):
                self.send_json({"error": "A checkpoint is already being processed"}, 409)
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if size <= 0:
                    raise ValueError("Choose a nonempty checkpoint file")
                workspace.state = {"state": "uploading"}
                name = unquote(self.headers.get("X-Weights-Name", "selected-weights.pt")).replace("\\", "/").rsplit("/", 1)[-1]
                if not name or name in {".", ".."}:
                    raise ValueError("Invalid checkpoint filename")
                inputs = workspace.root / "inputs"
                inputs.mkdir(exist_ok=True)
                upload = inputs / secrets.token_hex(8)
                upload.mkdir()
                weights = upload / name
                self.connection.settimeout(120)
                with weights.open("wb") as stream:
                    remaining = size
                    while remaining:
                        chunk = self.rfile.read(min(1024 * 1024, remaining))
                        if not chunk:
                            raise ValueError("Checkpoint upload was interrupted")
                        stream.write(chunk)
                        remaining -= len(chunk)
                workspace.start(weights)
                self.send_json({"state": "running"}, 202)
            except (OSError, ValueError) as error:
                workspace.state = {"state": "error", "error": str(error)}
                self.send_json(workspace.state, 400)
            finally:
                workspace.lock.release()

    return partial(SessionHandler, directory=str(workspace.public), service=workspace.service)


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights", nargs="?", type=Path, help="mini_trainer weights; omit to choose a file in the browser")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "mini-trainer" / "prototype-analysis",
        help="Persistent completed-analysis cache",
    )
    parser.add_argument("--clear-cache", action="store_true", help="Clear completed analysis cache entries and exit")
    parser.add_argument("--no-cache", action="store_true", help="Recompute analysis without reading or writing the cache")
    parser.add_argument("--export", action="store_true", help="Write a portable HTML report and exit")
    parser.add_argument("--output", type=Path, help="Export directory (default: <weights-stem>-explorer)")
    parser.add_argument("--pca-only", action="store_true", help="Skip the slower angular t-SNE projection")
    parser.add_argument("--synthetic", action="store_true", help="Include synthetic comparison cases")
    parser.add_argument("--threads", type=int, default=4, help="CPU analysis threads (default: 4)")
    parser.add_argument("--port", type=int, default=0, help="Local server port (default: choose an available port)")
    parser.add_argument("--no-browser", action="store_true", help="Print the local URL without opening a browser")
    args = parser.parse_args()
    if args.clear_cache:
        from .cache import clear_analysis_cache

        print(f"Removed {clear_analysis_cache(args.cache_dir)} analysis cache entries")
        return
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.weights is not None and not args.weights.is_file():
        parser.error(f"Checkpoint does not exist: {args.weights}")
    if args.export and args.weights is None:
        parser.error("--export requires a weights path")
    if args.output and not args.export:
        parser.error("--output is used with --export")
    try:
        require_dependencies(not args.pca_only)
    except RuntimeError as error:
        parser.error(str(error))
    if args.export:
        output = args.output or Path(args.weights.stem + "-explorer")
        generate(
            args.weights,
            output,
            angular=not args.pca_only,
            synthetic=args.synthetic,
            threads=args.threads,
            cache_dir=None if args.no_cache else args.cache_dir,
        )
        return

    from diskcache import Cache

    from mini_trainer.integrations import gbif

    from .serve import PhotoService

    with tempfile.TemporaryDirectory(prefix="mini-trainer-explorer-") as temporary:
        workspace = Workspace(Path(temporary), not args.pca_only, args.synthetic, args.threads, None if args.no_cache else args.cache_dir)
        with Cache(str(workspace.root / "photos"), size_limit=256 * 1024 * 1024) as cache:
            previous_cache = gbif._CACHE
            gbif._CACHE = cache
            workspace.service = PhotoService([], cache)
            server = ThreadingHTTPServer(("127.0.0.1", args.port), session_handler(workspace))
            if args.weights:
                workspace.start(args.weights.resolve())
            url = f"http://localhost:{server.server_port}/"
            print(f"Prototype explorer: {url}", flush=True)
            if not args.no_browser:
                webbrowser.open(url)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
            finally:
                server.server_close()
                workspace.close()
                gbif._CACHE = previous_cache
