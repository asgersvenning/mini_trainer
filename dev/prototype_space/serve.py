"""Serve the local prototype report with optional, cached GBIF reference photos."""

import argparse
import hashlib
import json
import re
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import urlopen

from diskcache import Cache

from mini_trainer.integrations import gbif


def web_url(value):
    return isinstance(value, str) and urlsplit(value).scheme in {"http", "https"}


def reference_photos(class_id, taxon, occurrences, limit=6):
    """Keep requested IDs, verify taxonomic membership, and retain media credits.

    These are GBIF examples, not images selected by model similarity. In
    particular, an occurrence's data license is not assumed to license its photo.
    """
    accepted = str(taxon.get("acceptedKey") or class_id)
    photos, seen = [], set()
    for occurrence in occurrences:
        membership = {
            str(occurrence.get(key))
            for key in (
                "taxonKey",
                "acceptedTaxonKey",
                "speciesKey",
                "genusKey",
                "familyKey",
                "orderKey",
                "classKey",
                "phylumKey",
                "kingdomKey",
            )
        }
        occurrence_id = str(occurrence.get("key", ""))
        if not {str(class_id), accepted}.intersection(membership) or not occurrence_id.isdigit():
            continue
        for media in occurrence.get("media", []):
            identifier = media.get("identifier")
            if media.get("type") != "StillImage" or not web_url(identifier) or identifier in seen:
                continue
            seen.add(identifier)
            digest = hashlib.md5(identifier.encode(), usedforsecurity=False).hexdigest()
            photos.append(
                {
                    "occurrence_id": occurrence_id,
                    "image_path": f"/gbif-image/{occurrence_id}/{digest}",
                    "original": identifier,
                    "source": media.get("references")
                    if web_url(media.get("references"))
                    else f"https://www.gbif.org/occurrence/{occurrence_id}",
                    "creator": media.get("creator") or media.get("rightsHolder") or "Creator not supplied",
                    "license": media.get("license") or "Image license not supplied",
                    "scientific_name": occurrence.get("scientificName", ""),
                }
            )
            if len(photos) >= limit:
                break
        if len(photos) >= limit:
            break
    return {
        "class_id": str(class_id),
        "display_name": taxon.get("canonicalName") or taxon.get("scientificName") or str(class_id),
        "accepted_id": accepted,
        "taxonomic_status": taxon.get("taxonomicStatus"),
        "photos": photos,
        "selection": "GBIF class examples",
    }


class PhotoService:
    def __init__(self, allowed_ids, cache):
        self.allowed_ids = set(allowed_ids)
        self.cache = cache
        self.image_paths = set()
        # GBIF asks scripted image-cache users to use a single HTTP connection.
        # Serialize all upstream requests, including image proxy requests.
        self.lock = threading.Lock()

    def metadata(self, class_id):
        if class_id not in self.allowed_ids or not class_id.isdigit():
            raise ValueError("Class is not a GBIF-ID candidate in this report")
        with self.lock:
            taxon = gbif.retrive_request(f"{gbif.GBIF_SPECIES_API_ENDPOINT}{class_id}")
            result = gbif.retrive_request(f"https://api.gbif.org/v1/occurrence/search?taxonKey={class_id}&mediaType=StillImage&limit=20")
            data = reference_photos(class_id, taxon, result.get("results", []))
            self.image_paths.update(photo["image_path"] for photo in data["photos"])
            return data

    def image(self, path):
        with self.lock:
            if path not in self.image_paths:
                raise ValueError("Image has not been resolved for a class in this report")
            cache_key = "prototype-image:" + path
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached
            occurrence, digest = path.split("/")[-2:]
            url = f"https://api.gbif.org/v1/image/cache/400x/occurrence/{occurrence}/media/{digest}"
            with urlopen(url, timeout=15) as response:
                content_type = response.headers.get_content_type()
                if content_type not in {"image/jpeg", "image/png", "image/webp", "image/gif"}:
                    raise ValueError("Upstream response was not a supported image")
                content = response.read(8 * 1024 * 1024 + 1)
            if len(content) > 8 * 1024 * 1024:
                raise ValueError("Thumbnail exceeded the 8 MiB limit")
            result = content, content_type
            self.cache.set(cache_key, result, expire=gbif.CACHE_TIME)
            return result


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, service, **kwargs):
        self.service = service
        super().__init__(*args, **kwargs)

    def send_json(self, payload, status=200):
        content = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self):
        path = urlsplit(self.path).path
        try:
            if path == "/api/health":
                return self.send_json({"photo_api": True})
            if path.startswith("/api/gbif/"):
                return self.send_json(self.service.metadata(path.removeprefix("/api/gbif/")))
            if re.fullmatch(r"/gbif-image/\d+/[a-f0-9]{32}", path):
                content, content_type = self.service.image(path)
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(content)))
                self.send_header("Cache-Control", "private, max-age=86400")
                self.end_headers()
                self.wfile.write(content)
                return None
            return super().do_GET()
        except (BrokenPipeError, ConnectionResetError):
            return None  # Navigation can cancel a queued photo request.
        except (ValueError, HTTPError, OSError, RuntimeError) as error:
            try:
                return self.send_json({"error": str(error)}, status=502)
            except (BrokenPipeError, ConnectionResetError):
                return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("tmp/prototype-report"))
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    directory = args.directory.resolve()
    data = json.loads((directory / "report-data.json").read_text())
    ids = {name for case in data.values() if not case["metadata"].get("synthetic") for name in case["names"] if name.isdigit()}
    del data
    # Reuse the repository request cache without writing into the user's shared
    # home cache. Keep it outside the directory exposed by the report server.
    cache = Cache(str(directory.with_name(directory.name + "-gbif-cache")), size_limit=256 * 1024 * 1024)
    gbif._CACHE = cache
    service = PhotoService(ids, cache)
    handler = partial(Handler, directory=str(directory), service=service)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    print(f"Prototype + optional GBIF photos: http://localhost:{args.port}/explorer.html", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        cache.close()


if __name__ == "__main__":
    main()
