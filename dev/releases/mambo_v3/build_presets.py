"""Build or verify release preset assets and their public scope catalogue."""

import argparse
import hashlib
import json
import tomllib
from pathlib import Path

from dev.releases.mambo_v3.audit import HERE, sha256

ROOT = HERE.parents[2]


def select_region(table, rule):
    """Union continent/country predicates, then apply explicit restrictions."""
    import pyarrow as pa
    import pyarrow.compute as pc

    permitted = {
        "label",
        "scope",
        "countries",
        "continents",
        "excluded_countries",
        "state_province",
        "country_state_restrictions",
        "country_continent_restrictions",
        "minimum_regional_rows",
        "minimum_global_rows",
        "minimum_latitude",
    }
    if unknown := set(rule) - permitted:
        raise ValueError(f"Unknown region fields: {sorted(unknown)}")
    mask = None
    for key, column in (("countries", "countryCode"), ("continents", "continent")):
        if values := rule.get(key):
            part = pc.fill_null(pc.is_in(table[column], value_set=pa.array(values)), False)
            mask = part if mask is None else pc.or_(mask, part)
    if "minimum_latitude" in rule:
        minimum = rule["minimum_latitude"]
        if not -90 <= minimum <= 90:
            raise ValueError("minimum_latitude must be within [-90, 90]")
        text = pc.utf8_trim_whitespace(pc.cast(table["decimalLatitude"], pa.string()))
        numeric = pc.match_substring_regex(text, r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")
        latitude = pc.cast(pc.if_else(numeric, text, None), pa.float64())
        northern = pc.fill_null(pc.and_(pc.greater_equal(latitude, minimum), pc.less_equal(latitude, 90)), False)
        mask = northern if mask is None else pc.and_(mask, northern)
    if mask is None:
        raise ValueError("Region requires countries, continents or minimum_latitude")
    if values := rule.get("excluded_countries"):
        mask = pc.and_(mask, pc.invert(pc.is_in(table["countryCode"], value_set=pa.array(values))))
    if values := rule.get("state_province"):
        mask = pc.and_(mask, pc.fill_null(pc.is_in(table["stateProvince"], value_set=pa.array(values)), False))
    for country, states in rule.get("country_state_restrictions", {}).items():
        is_country = pc.fill_null(pc.equal(table["countryCode"], country), False)
        in_state = pc.fill_null(pc.is_in(table["stateProvince"], value_set=pa.array(states)), False)
        mask = pc.and_(mask, pc.or_(pc.invert(is_country), in_state))
    for country, continents in rule.get("country_continent_restrictions", {}).items():
        is_country = pc.fill_null(pc.equal(table["countryCode"], country), False)
        in_continent = pc.fill_null(pc.is_in(table["continent"], value_set=pa.array(continents)), False)
        mask = pc.and_(mask, pc.or_(pc.invert(is_country), in_continent))
    return table.filter(mask)


def ordered_membership(counts, minimum, vocabulary, global_counts=None, global_minimum=0):
    if global_minimum and global_counts is None:
        raise ValueError("Global qualification requires global counts")
    chosen = {
        species
        for species, count in counts.items()
        if count >= minimum and (not global_minimum or global_counts.get(species, 0) >= global_minimum)
    }
    if unknown := chosen - set(vocabulary):
        raise ValueError(f"Selected species missing from model: {sorted(unknown)}")
    return [label for label in vocabulary if label in chosen]


def recipe(rule):
    parts = []
    for key, column in (("continents", "continent"), ("countries", "countryCode")):
        if rule.get(key):
            parts.append(f"`{column}` in `{', '.join(rule[key])}`")
    result = " OR ".join(parts)
    if "minimum_latitude" in rule:
        latitude = f"valid `decimalLatitude` between {rule['minimum_latitude']} and 90 degrees inclusive"
        result = f"({result}) AND {latitude}" if result else latitude
    if rule.get("excluded_countries"):
        result = f"({result}) AND country NOT in `{', '.join(rule['excluded_countries'])}`"
    if rule.get("state_province"):
        result = f"({result}) AND `stateProvince` in `{', '.join(rule['state_province'])}`"
    for country, states in rule.get("country_state_restrictions", {}).items():
        result += f"; `{country}` records additionally require `stateProvince` in `{', '.join(states)}`"
    for country, continents in rule.get("country_continent_restrictions", {}).items():
        result += f"; `{country}` records additionally require `continent` in `{', '.join(continents)}`"
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
    table = pq.read_table(metadata, columns=["speciesKey", "countryCode", "continent", "stateProvince", "decimalLatitude"])
    if table.num_rows != source["rows"] or table["speciesKey"].null_count:
        raise ValueError("Unexpected metadata rows or null species IDs")
    regional_minimum = definitions["minimum_regional_rows"]
    global_minimum = definitions["minimum_global_rows"]
    global_counts = {row["values"]: row["counts"] for row in table["speciesKey"].value_counts().to_pylist()}
    outputs = {}
    manifest = [
        "schema_version = 2",
        f'qualification_status = "{definitions["qualification_status"]}"',
        f'source_sha256 = "{source["sha256"]}"',
        f'model_manifest_sha256 = "{manifest_item["sha256"]}"',
        f'definitions_sha256 = "{sha256(definitions_path)}"',
        f"minimum_regional_rows = {regional_minimum}",
        f"minimum_global_rows = {global_minimum}",
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
        f"**Provisional qualification:** new presets require at least {regional_minimum} regional rows and "
        f"at least {global_minimum} global rows for a species. These inclusive thresholds are a working proposal, "
        "pending the final release decision. They reduce weak occurrence evidence but do not prove that records are independent "
        "or correctly geolocated: multiple images may belong to one observation. The global count measures available examples, "
        "not demonstrated model quality. Legacy presets retain their historical >25 regional-row rule with no new global gate.",
        "",
        f"In this pinned snapshot every model species has at least {min(global_counts.get(label, 0) for label in vocabulary)} "
        f"global rows; {sum(global_counts.get(label, 0) < global_minimum for label in vocabulary)} model species fall below "
        f"the proposed global minimum of {global_minimum}. "
        "Before finalizing qualification, decide whether regional evidence should count distinct GBIF observations instead of rows, "
        "and assess the effect on rare-species coverage. The present rule is reproducible, not a claim of ecological certainty.",
        "",
        "`europe` and `north_europe` preserve MAMBO_v2 membership while the V3 deployment default is global (`full`). "
        "Choose `europe_v3` or `north_europe_v3` for the new occurrence thresholds with the same explicit geographic filters. "
        "Parenthesized countries have ambiguous historical inclusion and leave the legacy list unchanged; "
        "that equivalence does not establish equivalence at the lower threshold, so they are not silently added. "
        "The deployment API discovers all lists from the bundle; Flemming evaluation favours legacy north_europe; "
        "updated membership remains an explicit broader option.",
        "",
        "## Presets",
        "",
        "| ID | Species | Minimum regional rows | Minimum global rows | Selected rows | Geographic scope |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
        f"| `full` | {len(vocabulary):,} | None | None | — | All species in the pinned model. |",
    ]
    summaries, memberships = {}, {}
    for name, rule in definitions["presets"].items():
        region_minimum = rule.get("minimum_regional_rows", regional_minimum)
        world_minimum = rule.get("minimum_global_rows", global_minimum)
        selected = select_region(table, rule)
        counts = {row["values"]: row["counts"] for row in selected["speciesKey"].value_counts().to_pylist()}
        labels = ordered_membership(counts, region_minimum, vocabulary, global_counts, world_minimum)
        regional_only = ordered_membership(counts, region_minimum, vocabulary)
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
                f"minimum_regional_rows = {region_minimum}",
                f"minimum_global_rows = {world_minimum}",
                f"excluded_by_global_gate_after_regional = {len(regional_only) - len(labels)}",
                f"selected_rows = {selected.num_rows}",
                f"species_before_threshold = {len(counts)}",
                f'sha256 = "{digest}"',
            ]
        )
        documentation.append(
            f"| `{name}` | {len(labels):,} | {region_minimum} | {world_minimum or 'None'} | {selected.num_rows:,} | {rule['scope']} |"
        )
        summaries[name] = len(labels)
        memberships[name] = labels
    changes = ["schema_version = 1", f'source_sha256 = "{source["sha256"]}"']
    documentation.extend(
        [
            "",
            "## Updated European presets",
            "",
            "| Updated ID | Legacy ID | Added species | Removed species |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for legacy in ("europe", "north_europe"):
        updated = f"{legacy}_v3"
        old_set, new_set = set(memberships[legacy]), set(memberships[updated])
        added = [label for label in memberships[updated] if label not in old_set]
        removed = [label for label in memberships[legacy] if label not in new_set]
        changes.extend(
            ["", f"[updates.{updated}]", f'legacy = "{legacy}"', f"added = {json.dumps(added)}", f"removed = {json.dumps(removed)}"]
        )
        documentation.append(f"| `{updated}` | `{legacy}` | {len(added)} | {len(removed)} |")
    documentation.extend(
        [
            "",
            "The [exact added/removed species IDs](../dev/releases/mambo_v3/preset-updates.toml) "
            "retain model order. Geographic filters are unchanged; only qualification thresholds differ.",
        ]
    )
    outputs[HERE / "preset-updates.toml"] = ("\n".join(changes) + "\n").encode()
    documentation.extend(
        [
            "",
            "## Species overlap",
            "",
            "![Pairwise species overlap and directional coverage](assets/preset-overlap.svg)",
            "",
            "Left: shared species divided by the union (Jaccard similarity). Right: the percentage of each row's species "
            "also present in each column. Coverage reveals containment that Jaccard can hide for small lists. "
            "These compare the qualified species lists, not geographic areas or prediction accuracy. Full is omitted because "
            "it contains every preset. Labels show list sizes; both panels use percentages.",
            "",
            "Rebuild the figure and exact shared-count/percentage table with "
            "`.venv/bin/python -m dev.releases.mambo_v3.plot_overlap`. "
            "The companion [pairwise table](assets/preset-overlap.tsv) includes exact counts.",
            "",
            "## Exact metadata filters",
            "",
        ]
    )
    for name, rule in definitions["presets"].items():
        documentation.extend([f"- **{name}** ({rule['label']}): {recipe(rule)}."])
    documentation.extend(
        [
            "",
            "## Interpretation and reproducibility",
            "",
            "Mexico belongs to North and Central America; Costa Rica and Panama belong to Central and South America. "
            "Australia includes Tasmania; Tasmania-only uses the explicit state field and does not mean endemic-only. "
            "Arctic uses latitude at least 60°N across all countries, including the boundary; "
            "missing, malformed or out-of-range latitudes are excluded. This broad northern scope includes subarctic areas. "
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
