"""Resume publication from retained artifacts, never by rebuilding a published version."""

import argparse
import json
import os
import subprocess
from pathlib import Path


def validate_run(run, jobs, release, tag_commit):
    if (
        run["event"] != "release"
        or run["path"] != ".github/workflows/publish-model.yml"
        or run["status"] != "completed"
        or run["head_sha"] != tag_commit
        or release["draft"]
        or release["prerelease"]
        or not release.get("published_at")
    ):
        raise ValueError("Recovery requires a completed publication run for the unchanged public release tag")
    succeeded = {job["name"] for job in jobs if job["conclusion"] == "success"}
    if not {"prepare", "package"} <= succeeded:
        raise ValueError("Recovery requires successful preparation and package publication")
    return run["head_sha"]


def validate_payload(folder, source, product, version):
    from dev.releases.mambo_v3.publish_assets import verified_payload

    for name in ("github", "model", "space"):
        payload = verified_payload(folder / name)
        if payload["source_commit"] != source or payload["package_version"] != version:
            raise ValueError("Retained publication identity differs from the original release")
    candidate = json.loads((folder / "github/release-candidate.json").read_text())
    if candidate["distribution"] != product or candidate["qualification"] != "passed":
        raise ValueError("Retained candidate is not the qualified release product")
    for name in ("github", "model", "space"):
        payload = json.loads((folder / name / "publication.json").read_text())
        if any(payload[key] != candidate[key] for key in ("model_id", "package_version", "source_commit")):
            raise ValueError("Retained candidate and publication identities differ")


def metadata(repository, run_id, tag):
    def get(path):
        return json.loads(subprocess.check_output(["gh", "api", f"repos/{repository}/{path}"], text=True))

    run = get(f"actions/runs/{run_id}")
    jobs = get(f"actions/runs/{run_id}/jobs?per_page=100")["jobs"]
    release = get(f"releases/tags/{tag}")
    return validate_run(run, jobs, release, get(f"commits/{tag}")["sha"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--run", type=int, required=True)
    run_parser.add_argument("--tag", required=True)
    payload_parser = sub.add_parser("payload")
    payload_parser.add_argument("folder", type=Path)
    payload_parser.add_argument("--source", required=True)
    payload_parser.add_argument("--product", required=True)
    payload_parser.add_argument("--version", required=True)
    args = parser.parse_args()
    if args.command == "run":
        source = metadata(os.environ["GITHUB_REPOSITORY"], args.run, args.tag)
        with Path(os.environ["GITHUB_OUTPUT"]).open("a") as stream:
            stream.write(f"source={source}\n")
    else:
        validate_payload(args.folder, args.source, args.product, args.version)
