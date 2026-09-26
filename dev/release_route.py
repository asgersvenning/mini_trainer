"""Resolve release intent from branch/tag identity; never authorize publication on push."""

import argparse
import json
import os
import re
import tomllib
from pathlib import Path


def products(root, kind):
    if kind == "packages":
        project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
        return {project["name"]: {"project": "."}}
    return tomllib.loads((root / ".github/model-releases.toml").read_text())["models"]


def resolve(root, kind, event_name, ref, event, selected=None):
    entries = products(root, kind)
    tag = event.get("release", {}).get("tag_name", "")
    category = "models" if kind == "demos" else kind
    if event_name == "release":
        if kind == "demos":
            return {"enabled": "false"}
        prefix = f"{category}/"
        if tag.startswith(prefix):
            parts = tag.split("/")
            if len(parts) != 3:
                raise ValueError("Expected category/product/vVERSION release tag")
            selected = parts[1]
        else:
            selected = next((name for name, item in entries.items() if item.get("tag") == tag), None)
            if selected is None:
                return {"enabled": "false"}
    elif event_name == "push":
        prefix = f"refs/heads/release/{kind}/"
        if not ref.startswith(prefix):
            return {"enabled": "false"}
        selected = ref.removeprefix(prefix)
    elif event_name == "workflow_dispatch":
        if kind == "packages" and not selected:
            selected = next(iter(entries))
    else:
        return {"enabled": "false"}
    if selected not in entries or not re.fullmatch(r"[a-z0-9][a-z0-9-]*", selected):
        raise ValueError(f"Unknown {kind} product: {selected!r}")
    entry = entries[selected]
    project_path = (root / entry["project"] / "pyproject.toml").resolve()
    if not project_path.is_relative_to(root.resolve()):
        raise ValueError("Project must be inside the repository")
    project = tomllib.loads(project_path.read_text())["project"]
    if project["name"] != selected:
        raise ValueError("Release product differs from distribution name")
    version = project["version"]
    expected = entry.get("tag", f"{category}/{selected}/v{version}")
    if "tag" in entry and entry.get("tag_version") != version:
        raise ValueError("Explicit release tag must be reviewed for the new package version")
    if event_name == "release" and tag != expected:
        raise ValueError(f"Tag does not match checked-out release: expected {expected!r}")
    module = entry.get("module", "")
    if kind != "packages" and not re.fullmatch(r"[a-z_]\w*(?:\.[a-z_]\w*)+", module):
        raise ValueError("Model needs a valid preparation module")
    return {"enabled": "true", "product": selected, "version": version, "tag": expected, "module": module}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("packages", "models", "demos"))
    args = parser.parse_args()
    result = resolve(
        Path(__file__).resolve().parents[1],
        args.kind,
        os.environ["GITHUB_EVENT_NAME"],
        os.environ["GITHUB_REF"],
        json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text()),
        os.environ.get("RELEASE_PRODUCT"),
    )
    with Path(os.environ["GITHUB_OUTPUT"]).open("a") as stream:
        for key, value in result.items():
            if "\n" in value or "\r" in value:
                raise ValueError("Invalid multiline release metadata")
            stream.write(f"{key}={value}\n")
