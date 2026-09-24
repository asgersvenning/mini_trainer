"""Freeze a verified local bundle's small metadata for automatic ERDA bootstrap."""

import argparse
import hashlib
import json
from pathlib import Path

from deployment.mambo_deploy.bundle import Bundle


def package(source, output):
    bundle = Bundle(source)
    metadata = {}
    for relative in bundle.manifest["files"]:
        if relative not in bundle.manifest["origins"]:
            metadata[relative] = bundle.file(relative).read_text()
    # Ship current integration guidance, not the README frozen in the local bundle.
    readme = Path(__file__).resolve().parents[3] / "deployment/README.md"
    metadata["README.md"] = readme.read_text()
    data = metadata["README.md"].encode()
    bundle.manifest["files"]["README.md"] = {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    metadata["release.json"] = json.dumps(bundle.manifest, indent=2) + "\n"
    output.write_text(json.dumps({"metadata": metadata}, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    package(args.source, args.output)
