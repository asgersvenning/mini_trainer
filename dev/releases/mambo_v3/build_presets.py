"""Build or verify release preset assets and their public scope catalogue."""

import argparse
import hashlib
import json
import tomllib
from pathlib import Path

from dev.releases.mambo_v3.audit import HERE, sha256
from dev.releases.mambo_v3.reconstruct_presets import membership

ROOT = HERE.parents[2]


def select_region(table, rule):
    """Union continent/country predicates, then apply explicit restrictions."""
    import pyarrow as pa
    import pyarrow.compute as pc

    permitted = {"label", "scope", "countries", "continents", "excluded_countries", "state_province", "exclusive_minimum_rows"}
    if unknown := set(rule) - permitted:
        raise ValueError(f"Unknown region fields: {sorted(unknown)}")
    mask = None
    for key, column in (("countries", "countryCode"), ("continents", "continent")):
        if values := rule.get(key):
            part = pc.fill_null(pc.is_in(table[column], value_set=pa.array(values)), False)
            mask = part if mask is None else pc.or_(mask, part)
    if mask is None:
        raise ValueError("Region requires countries or continents")
    if values := rule.get("excluded_countries"):
        mask = pc.and_(mask, pc.invert(pc.is_in(table["countryCode"], value_set=pa.array(values))))
    if values := rule.get("state_province"):
        mask = pc.and_(mask, pc.fill_null(pc.is_in(table["stateProvince"], value_set=pa.array(values)), False))
    return table.filter(mask)


def ordered_membership(counts, threshold, vocabulary):
    chosen = membership(counts, threshold)
    if unknown := chosen - set(vocabulary):
        raise ValueError(f"Selected species missing from model: {sorted(unknown)}")
    return [label for label in vocabulary if label in chosen]


def recipe(rule):
    parts = []
    for key, column in (("continents", "continent"), ("countries", "countryCode")):
        if rule.get(key):
            parts.append(f"`{column}` in `{', '.join(rule[key])}`")
    result = " OR ".join(parts)
    if rule.get("excluded_countries"):
        result = f"({result}) AND country NOT in `{', '.join(rule['excluded_countries'])}`"
    if rule.get("state_province"):
        result = f"({result}) AND `stateProvince` in `{', '.join(rule['state_province'])}`"
    return result


def build(metadata, evidence_root, write=False):
    import pyarrow.parquet as pq

    construction = tomllib.loads((HERE / "construction.toml").read_text())
    source = construction["source"]
    if metadata.stat().st_size != source["size"] or sha256(metadata) != source["sha256"]:
        raise ValueError("Source Parquet differs from pinned metadata")
    definitions_path = HERE / "preset-definitions.toml"
    definitions = tomllib.loads(definitions_path.read_text())
    inventory = tomllib.loads((HERE / "inventory.toml").read_text())
    manifest_item = next(item for item in inventory["artifacts"] if item["path"].endswith("models/onnx-fp32/manifest.json"))
    model_manifest = evidence_root / manifest_item["path"]
    if sha256(model_manifest) != manifest_item["sha256"]:
        raise ValueError("Model manifest differs from pinned source")
    mapping = json.loads(model_manifest.read_text())["classifiers"][0]["metadata"]["cls2idx"]["0"]
    if sorted(mapping.values()) != list(range(len(mapping))):
        raise ValueError("Model species indices are not contiguous")
    vocabulary = sorted(mapping, key=mapping.get)
    table = pq.read_table(metadata, columns=["speciesKey", "countryCode", "continent", "stateProvince"])
    if table.num_rows != source["rows"] or table["speciesKey"].null_count:
        raise ValueError("Unexpected metadata rows or null species IDs")
    threshold = definitions["exclusive_minimum_rows"]
    outputs = {}
    manifest = [
        "schema_version = 1",
        f'source_sha256 = "{source["sha256"]}"',
        f'model_manifest_sha256 = "{manifest_item["sha256"]}"',
        f'definitions_sha256 = "{sha256(definitions_path)}"',
        f"exclusive_minimum_rows = {threshold}",
    ]
    documentation = [
        "# Model preset scope",
        "",
        "Generated release catalogue: edit `dev/releases/mambo_v3/preset-definitions.toml`, then use the build command below.",
        "",
        "These overlapping deployment presets aim to avoid most geographically nonsensical predictions while allowing species "
        "that **can be found** in a region. They do not describe native or natural distributions, or where species should occur. "
        "Recorded introduced species, migrants and vagrants are eligible: no native-status or establishment filter is applied. "
        "Exclusion is not evidence that a species cannot occur there. Country codes are ISO alpha-2 metadata values.",
        "",
        "Each geographic preset applies the minimum row count shown below. "
        "Counts use all existing splits, including held-out rows, without further deduplication. "
        "Each row counts once even if it matches both a country and a continent predicate. "
        "Full uses all model species without a regional threshold. Lists retain the model's species order.",
        "",
        "Europe and northern Europe preserve MAMBO_v2 membership. Parenthesized countries in northern Europe's scope "
        "have ambiguous historical inclusion and do not change its membership. The other presets are new release definitions. "
        "These are release assets; adapter/API discovery integration and preset-specific inference qualification are still pending.",
        "",
        "## Presets",
        "",
        "| ID | Species | Minimum rows per species | Selected rows | Geographic scope |",
        "| --- | ---: | ---: | ---: | --- |",
        f"| `full` | {len(vocabulary):,} | — | — | All species in the pinned model. |",
    ]
    summaries = {}
    for name, rule in definitions["presets"].items():
        region_threshold = rule.get("exclusive_minimum_rows", threshold)
        selected = select_region(table, rule)
        counts = {row["values"]: row["counts"] for row in selected["speciesKey"].value_counts().to_pylist()}
        labels = ordered_membership(counts, region_threshold, vocabulary)
        if not labels:
            raise ValueError(f"Empty preset: {name}")
        data = ("\n".join(labels) + "\n").encode()
        digest = hashlib.sha256(data).hexdigest()
        if name in inventory["presets"] and digest != inventory["presets"][name]["sha256"]:
            raise ValueError(f"Legacy preset changed: {name}")
        outputs[HERE / "presets" / f"{name}.classes"] = data
        manifest.extend(
            [
                "",
                f"[presets.{name}]",
                f'path = "presets/{name}.classes"',
                f"count = {len(labels)}",
                f"exclusive_minimum_rows = {region_threshold}",
                f"selected_rows = {selected.num_rows}",
                f"species_before_threshold = {len(counts)}",
                f'sha256 = "{digest}"',
            ]
        )
        documentation.append(f"| `{name}` | {len(labels):,} | {region_threshold + 1} | {selected.num_rows:,} | {rule['scope']} |")
        summaries[name] = len(labels)
    documentation.extend(["", "## Exact metadata filters", ""])
    for name, rule in definitions["presets"].items():
        documentation.extend([f"- **{name}** ({rule['label']}): {recipe(rule)}."])
    documentation.extend(
        [
            "",
            "## Interpretation and reproducibility",
            "",
            "Mexico belongs to North and Central America; Costa Rica and Panama belong to Central and South America. "
            "Australia includes Tasmania; Tasmania-only uses the explicit state field and does not mean endemic-only. "
            "Arctic is a broad northern-country proxy and includes southern records from those countries. "
            "Regional restrictions change score normalization; excluded truth labels must remain visible in evaluation.",
            "",
            "Blank geographic fields match no predicate unless another selected field matches. The Tasmania preset excludes "
            "Australian records with blank or different state values. "
            "Overlapping presets are expected; membership in one does not exclude another.",
            "",
            "Run from the repository root with the existing PyArrow environment and the previously downloaded model manifest:",
            "",
            "```sh",
            ".venv/bin/python -m dev.releases.mambo_v3.build_presets \\",
            f"  {source['path']} \\",
            "  --evidence-root local-evidence/mambo-v3",
            "```",
            "",
            "The default checks committed assets, hashes, counts and this catalogue against fresh reconstruction. "
            "Use `--write` after an intentional definition update to regenerate them. Legacy preset hashes must remain unchanged. "
            "Published preset membership changes require a new release/revision and an added/removed-ID report.",
            "",
            "[Machine-readable definitions](../dev/releases/mambo_v3/preset-definitions.toml), "
            "[generated hashes and counts](../dev/releases/mambo_v3/preset-manifest.toml), "
            "[source provenance](../dev/releases/mambo_v3/construction.toml), "
            "[legacy reconstruction details](../dev/releases/mambo_v3/README.md#regional-scope-and-construction).",
            "",
        ]
    )
    outputs[HERE / "preset-manifest.toml"] = ("\n".join(manifest) + "\n").encode()
    outputs[ROOT / "docs/model-presets.md"] = "\n".join(documentation).encode()
    for path, data in outputs.items():
        if write:
            path.write_bytes(data)
        elif not path.exists() or path.read_bytes() != data:
            raise ValueError(f"Preset output missing or stale: {path}")
    return summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metadata", type=Path)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build(args.metadata, args.evidence_root, args.write), indent=2))
