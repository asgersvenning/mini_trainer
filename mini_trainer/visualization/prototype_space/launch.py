"""Launch the prototype explorer with a checkpoint path or browser file picker."""

import argparse
import importlib.util
import json
import secrets
import subprocess
import sys
import tempfile
import threading
import webbrowser
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit


def require_dependencies(angular=True):
    missing = [name for name in (["scipy", "sklearn"] if angular else ["scipy"]) if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError("Prototype exploration requires: " + ", ".join(missing) + ". Install mini_trainer[explorer].")


def generate(weights, output, *, angular=True, synthetic=False, threads=4):
    require_dependencies(angular)
    import torch

    from .explore import create_report

    torch.set_num_threads(threads)
    return create_report(weights, output, include_tsne=angular, synthetic=synthetic)


class Workspace:
    """One local session; checkpoint bytes are never in the served directory."""

    def __init__(self, root, angular=True, synthetic=False, threads=4):
        self.root = root
        self.public = root / "report"
        self.public.mkdir()
        self.angular, self.synthetic, self.threads = angular, synthetic, threads
        self.token = secrets.token_urlsafe(32)
        self.lock = threading.Lock()
        self.process = None
        self.state = {"state": "idle"}
        self.service = None

    def start(self, weights):
        self.state = {"state": "running", "file": weights.name}
        worker = threading.Thread(target=self.build, args=(weights,), daemon=True)
        worker.start()

    def build(self, weights):
        try:
            command = [
                sys.executable,
                "-m",
                "mini_trainer.visualization.prototype_space",
                str(weights),
                "--export",
                "--output",
                str(self.public),
                "--threads",
                str(self.threads),
            ]
            if not self.angular:
                command.append("--pca-only")
            if self.synthetic:
                command.append("--synthetic")
            with (self.root / "generation.log").open("w") as log:
                self.process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                result = self.process.wait()
            if result:
                detail = (self.root / "generation.log").read_text()[-4000:]
                self.state = {"state": "error", "error": detail}
                return
            data = json.loads((self.public / "report-data.json").read_text())
            self.service.allowed_ids = {
                name for case in data.values() if not case["metadata"].get("synthetic") for name in case["names"] if name.isdigit()
            }
            self.state = {"state": "ready", "file": weights.name, "url": "/explorer.html"}
        except Exception as error:
            self.state = {"state": "error", "error": str(error)}
        finally:
            self.lock.release()


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
            if path == "/api/session":
                self.send_json(workspace.state)
                return
            return super().do_GET()

        def do_POST(self):
            if self.path != "/api/weights" or self.headers.get("X-Explorer-Token") != workspace.token:
                self.send_json({"error": "Invalid local session request"}, 403)
                return
            if not workspace.lock.acquire(blocking=False):
                self.send_json({"error": "A checkpoint is already being processed"}, 409)
                return
            handed_off = False
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
                weights = inputs / name
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
                handed_off = True
                self.send_json({"state": "running"}, 202)
            except (OSError, ValueError) as error:
                workspace.state = {"state": "error", "error": str(error)}
                self.send_json(workspace.state, 400)
            finally:
                if not handed_off:
                    workspace.lock.release()

    return partial(SessionHandler, directory=str(workspace.public), service=workspace.service)


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights", nargs="?", type=Path, help="mini_trainer weights; omit to choose a file in the browser")
    parser.add_argument("--export", action="store_true", help="Write a portable HTML report and exit")
    parser.add_argument("--output", type=Path, help="Export directory (default: <weights-stem>-explorer)")
    parser.add_argument("--pca-only", action="store_true", help="Skip the slower angular t-SNE projection")
    parser.add_argument("--synthetic", action="store_true", help="Include synthetic comparison cases")
    parser.add_argument("--threads", type=int, default=4, help="CPU analysis threads (default: 4)")
    parser.add_argument("--port", type=int, default=0, help="Local server port (default: choose an available port)")
    parser.add_argument("--no-browser", action="store_true", help="Print the local URL without opening a browser")
    args = parser.parse_args()
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
        generate(args.weights, output, angular=not args.pca_only, synthetic=args.synthetic, threads=args.threads)
        return

    from diskcache import Cache

    from mini_trainer.integrations import gbif

    from .serve import PhotoService

    with tempfile.TemporaryDirectory(prefix="mini-trainer-explorer-") as temporary:
        workspace = Workspace(Path(temporary), not args.pca_only, args.synthetic, args.threads)
        with Cache(str(workspace.root / "photos"), size_limit=256 * 1024 * 1024) as cache:
            previous_cache = gbif._CACHE
            gbif._CACHE = cache
            workspace.service = PhotoService([], cache)
            server = ThreadingHTTPServer(("127.0.0.1", args.port), session_handler(workspace))
            if args.weights:
                workspace.lock.acquire()
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
                if workspace.process is not None and workspace.process.poll() is None:
                    workspace.process.terminate()
                    workspace.process.wait()
                gbif._CACHE = previous_cache
