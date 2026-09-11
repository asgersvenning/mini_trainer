"""Prepare a resumable, optional species-name snapshot for a static report."""

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True, help="Exported report-data.json")
    parser.add_argument("--output", type=Path, required=True, help="Destination gbif-snapshot.json; resumes existing snapshots")
    parser.add_argument("--limit", type=int, default=0, help="Maximum new lookups; zero means all missing IDs")
    args = parser.parse_args()
    payload = json.loads(args.report.read_text())
    ids = set()
    for case in payload.values():
        if case["metadata"].get("synthetic"):
            continue
        ids.update(str(name) for name in case["names"])
        for group in case.get("groups", []):
            ids.update(str(name) for name in group)
    ids = sorted(name for name in ids if name.isdigit())
    snapshot = (
        json.loads(args.output.read_text())
        if args.output.exists()
        else {
            "schema": "mini-trainer-gbif-v1",
            "source": "https://api.gbif.org/v1/species/",
            "taxa": {},
            "retrieved_at": {},
        }
    )
    if snapshot.get("schema") != "mini-trainer-gbif-v1":
        raise ValueError("Unsupported snapshot schema")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    missing = [name for name in ids if name not in snapshot["taxa"]]
    if args.limit:
        missing = missing[: args.limit]
    print(f"{len(ids)} IDs; {len(missing)} new lookups; existing entries retained", flush=True)
    for index, name in enumerate(missing):
        try:
            request = Request("https://api.gbif.org/v1/species/" + name, headers={"User-Agent": "mini-trainer-prototype-snapshot/1"})
            with urlopen(request, timeout=15) as response:
                taxon = json.load(response)
            if str(taxon.get("key")) != name:
                raise ValueError("Returned taxon ID differs")
            snapshot["taxa"][name] = {
                key: taxon[key]
                for key in ("key", "canonicalName", "scientificName", "acceptedKey", "rank", "taxonomicStatus")
                if key in taxon
            }
            snapshot["retrieved_at"][name] = datetime.now(UTC).isoformat()
        except Exception as error:
            print(f"{name}: {error}; left missing for browser lookup or later retry", flush=True)
        temporary = args.output.with_suffix(".partial")
        temporary.write_text(json.dumps(snapshot, ensure_ascii=False) + "\n")
        temporary.replace(args.output)
        if (index + 1) % 50 == 0:
            print(f"{index + 1}/{len(missing)} attempted; {len(snapshot['taxa'])} cached", flush=True)
        time.sleep(0.1)
    print(f"Snapshot: {args.output}; {len(snapshot['taxa'])} taxa", flush=True)


if __name__ == "__main__":
    main()
