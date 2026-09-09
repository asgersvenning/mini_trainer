"""Restore compact history from monthly GitHub draft releases; optionally append records."""

import json
import os
import re
import subprocess
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

from .report_history import IDENTIFIER, REVISION, render

ARCHIVE_TAG = re.compile(r"benchmark-history-\d{4}-(?:0[1-9]|1[0-2])")
REPOSITORY = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")


class GitHub:
    """Use gh's authentication and pagination without passing tokens in arguments."""

    def __init__(self, repository):
        if not REPOSITORY.fullmatch(repository) or any(part in (".", "..") for part in repository.split("/")):
            raise ValueError("Require a GitHub owner/repository name")
        if os.environ.get("GH_HOST", "github.com") != "github.com":
            raise ValueError("History storage currently supports github.com only")
        self.root = f"repos/{repository}/releases"

    def api(self, endpoint, *args, payload=None):
        command = ["gh", "api", "--hostname", "github.com", endpoint, *args]
        if payload is not None:
            command += ["--input", "-"]
        try:
            result = subprocess.run(command, input=payload, capture_output=True, check=True)
        except FileNotFoundError as exc:
            raise RuntimeError("Install GitHub CLI (gh) and configure GH_TOKEN before accessing report storage") from exc
        except subprocess.CalledProcessError as exc:
            # Do not echo CLI stderr: external authentication helpers may include secrets.
            raise RuntimeError(f"GitHub request failed with exit code {exc.returncode}; archive was not overwritten") from None
        return result.stdout

    def listing(self, endpoint):
        pages = json.loads(self.api(endpoint + "?per_page=100", "--paginate", "--slurp"))
        return [item for page in pages for item in page]

    def download(self, asset):
        if type(asset["id"]) is not int or asset["id"] <= 0:
            raise ValueError("Invalid release asset identifier")
        return self.api(f"{self.root}/assets/{asset['id']}", "-H", "Accept: application/octet-stream")

    def create(self, tag, revision):
        payload = {
            "tag_name": tag,
            "target_commitish": revision,
            "name": tag,
            "body": "Compact benchmark records. Keep this storage release in draft; publish the derived history separately.",
            "draft": True,
            "prerelease": True,
            "make_latest": "false",
        }
        return json.loads(self.api(self.root, "--method", "POST", payload=json.dumps(payload).encode()))

    def upload(self, release, name, payload):
        endpoint = f"https://uploads.github.com/{self.root}/{release['id']}/assets?name={name}"
        return json.loads(self.api(endpoint, "--method", "POST", "-H", "Content-Type: application/json", payload=payload))


def record_identity(name, payload):
    record = json.loads(payload)
    run_id = record["run_id"]
    if not IDENTIFIER.fullmatch(run_id) or name != f"{run_id}.json":
        raise ValueError("History filename and run identity differ")
    if record.get("schema_version") != 1 or record.get("kind") != "tensorrt_deployment":
        raise ValueError("Unsupported compact history schema")
    if not REVISION.fullmatch(record["revision"]):
        raise ValueError("History record requires a full source revision")
    timestamp = datetime.fromisoformat(record["recorded_at"])
    if timestamp.utcoffset() is None or timestamp.utcoffset().total_seconds() != 0:
        raise ValueError("History timestamps must use UTC")
    return f"benchmark-history-{timestamp:%Y-%m}", record["revision"]


def synchronize(repository, output, records=None, upload=False):
    """Reconstruct a fresh local archive before making any optional remote writes.

    Existing bytes are immutable to this client. Maintainers can still delete or
    replace remote assets; keep an independent backup for stronger guarantees.
    """
    github = GitHub(repository)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    destination = output / "records"
    destination.mkdir()
    releases, stored = {}, {}
    for release in github.listing(github.root):
        tag = release["tag_name"]
        if not ARCHIVE_TAG.fullmatch(tag):
            continue
        if tag in releases:
            raise ValueError("Duplicate monthly history release")
        releases[tag] = release
        for asset in github.listing(f"{github.root}/{release['id']}/assets"):
            name = asset["name"]
            payload = github.download(asset)
            expected_tag, _ = record_identity(name, payload)
            if expected_tag != tag or name in stored:
                raise ValueError("Duplicate or misplaced remote history record")
            stored[name] = payload
            (destination / name).write_bytes(payload)
    incoming = {}
    if records is not None:
        source = Path(records)
        if not source.is_dir():
            raise ValueError("Incoming records directory does not exist")
        for path in sorted(source.glob("*.json")):
            payload = path.read_bytes()
            tag, revision = record_identity(path.name, payload)
            if path.name in stored:
                if stored[path.name] != payload:
                    raise ValueError("Run identity already exists with different bytes")
                continue
            incoming[path.name] = (payload, tag, revision)
            (destination / path.name).write_bytes(payload)
    # Validate and render the entire merged archive before creating releases/assets.
    render(output)
    if upload:
        for _, tag, _ in incoming.values():
            if tag in releases and (releases[tag].get("draft") is not True or releases[tag].get("immutable")):
                raise ValueError("Refusing to append to a published or immutable history release")
        for name, (payload, tag, revision) in incoming.items():
            if tag not in releases:
                releases[tag] = github.create(tag, revision)
            asset = github.upload(releases[tag], name, payload)
            if github.download(asset) != payload:
                raise ValueError("Uploaded history record failed byte-for-byte readback")
    return output / "index.html"


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--output", required=True, help="New directory for restored records and HTML")
    parser.add_argument("--records", help="Directory containing incoming compact JSON records")
    parser.add_argument("--upload", action="store_true", help="Append new records to remote draft release storage")
    synchronize(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
