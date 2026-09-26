"""Legacy reconstruction and deliberately overlapping deployment region contracts."""

import tomllib

import pytest

from dev.releases.mambo_v3.audit import HERE
from dev.releases.mambo_v3.build_presets import ordered_membership, select_region
from dev.releases.mambo_v3.reconstruct_presets import membership, region_counts

pa = pytest.importorskip("pyarrow")
DEFINITIONS = tomllib.loads((HERE / "preset-definitions.toml").read_text())
RULES = DEFINITIONS["presets"]


def test_continent_is_not_inferred_from_country_and_threshold_is_strict():
    table = pa.table(
        {
            "speciesKey": ["continental"] * 26 + ["boundary"] * 25 + ["island"] * 30,
            "countryCode": ["ES"] * 81,
            "continent": ["EUROPE"] * 51 + ["AFRICA"] * 30,
        }
    )
    counts = region_counts(table, "continent", ["EUROPE"])
    assert counts == {"continental": 26, "boundary": 25}
    assert membership(counts, 25) == {"continental"}
    assert membership(region_counts(table, "countryCode", ["ES"]), 25) == {"continental", "island"}


def test_country_union_counts_rows_across_splits_without_deduplication():
    table = pa.table(
        {
            "speciesKey": ["shared", "shared", "shared", "outside"],
            "countryCode": ["DE", "NL", "DE", "GB"],
            "set": ["0", "1", "1", "0"],
            "gbifID": ["same", "other", "same", "third"],
        }
    )
    assert region_counts(table, "countryCode", ["DE", "NL"]) == {"shared": 3}


def test_missing_species_identity_fails():
    table = pa.table({"speciesKey": pa.array([None], type=pa.string()), "countryCode": ["DE"]})
    with pytest.raises(ValueError, match="null species"):
        region_counts(table, "countryCode", ["DE"])


@pytest.mark.parametrize("legacy", ["europe", "north_europe"])
def test_updated_european_lists_preserve_geography_and_legacy_membership(legacy):
    updated = f"{legacy}_v3"
    descriptive = {"label", "scope", "minimum_regional_rows", "minimum_global_rows"}
    assert {k: v for k, v in RULES[legacy].items() if k not in descriptive} == {
        k: v for k, v in RULES[updated].items() if k not in descriptive
    }
    assert RULES[updated].get("minimum_regional_rows", DEFINITIONS["minimum_regional_rows"]) == 3
    assert RULES[updated].get("minimum_global_rows", DEFINITIONS["minimum_global_rows"]) == 25
    old = (HERE / "presets" / f"{legacy}.classes").read_text().splitlines()
    new = (HERE / "presets" / f"{updated}.classes").read_text().splitlines()
    old_members = set(old)
    assert old_members < set(new)
    assert [label for label in new if label in old_members] == old
    changes = tomllib.loads((HERE / "preset-updates.toml").read_text())["updates"][updated]
    assert changes["added"] == [label for label in new if label not in old_members]
    assert changes["removed"] == []


def countries_selected(name, countries):
    table = pa.table({"countryCode": countries, "continent": [""] * len(countries)})
    return select_region(table, RULES[name])["countryCode"].to_pylist()


def test_american_regions_intentionally_overlap():
    countries = ["MX", "CR", "PA", "CA", "BR"]
    assert countries_selected("north_america", countries) == ["MX", "CA"]
    assert countries_selected("central_america", countries) == ["MX", "CR", "PA"]
    assert countries_selected("south_america", countries) == ["CR", "PA", "BR"]


def test_tasmania_is_australian_state_not_endemism_filter():
    table = pa.table(
        {
            "countryCode": ["AU", "AU", "AU", "AU", "NZ"],
            "stateProvince": ["Tasmania", "Victoria", "", None, "Tasmania"],
            "speciesKey": ["widespread"] * 5,
        }
    )
    assert select_region(table, RULES["australia"]).num_rows == 4
    selected = select_region(table, RULES["tasmania"])
    assert selected.num_rows == 1
    assert selected["speciesKey"].to_pylist() == ["widespread"]


def test_union_does_not_duplicate_rows_and_exclusion_overrides_continent():
    table = pa.table({"countryCode": ["ZA", "EG", "SD", ""], "continent": ["AFRICA", "AFRICA", "", None]})
    assert select_region(table, RULES["africa"]).num_rows == 3
    assert select_region(table, RULES["subsaharan_africa"])["countryCode"].to_pylist() == ["ZA", "SD"]


def test_membership_preserves_model_order_and_rejects_unknown_species():
    assert ordered_membership({"a": 26, "b": 25, "c": 80}, 26, ["c", "b", "a"]) == ["c", "a"]
    assert ordered_membership({"a": 1, "b": 0}, 1, ["b", "a"]) == ["a"]
    with pytest.raises(ValueError, match="missing from model"):
        ordered_membership({"unknown": 26}, 25, ["a"])


def test_qualification_requires_both_inclusive_minima():
    regional = {"boundary": 3, "few_local": 2, "few_global": 10, "strong": 5, "missing": 3}
    worldwide = {"boundary": 25, "few_local": 100, "few_global": 24, "strong": 50}
    vocabulary = ["strong", "boundary", "few_local", "few_global", "missing"]
    assert ordered_membership(regional, 3, vocabulary, worldwide, 25) == ["strong", "boundary"]
    with pytest.raises(ValueError, match="requires global counts"):
        ordered_membership(regional, 3, vocabulary, global_minimum=25)


def test_misspelled_or_empty_filter_fails_closed():
    table = pa.table({"countryCode": ["AU"]})
    with pytest.raises(ValueError, match="Unknown region fields"):
        select_region(table, {"countries": ["AU"], "state_provinc": ["Tasmania"]})
    with pytest.raises(ValueError, match="requires countries, continents or minimum_latitude"):
        select_region(table, {})


def test_asia_excludes_cyprus_even_when_continent_matches():
    table = pa.table({"countryCode": ["CY", "JP", "TR"], "continent": ["ASIA", "ASIA", "EUROPE"]})
    assert select_region(table, RULES["asia"])["countryCode"].to_pylist() == ["JP", "TR"]


def test_arctic_uses_inclusive_latitude_independent_of_country_or_state():
    table = pa.table(
        {
            "countryCode": ["US", "US", "RU", "CA", "", "NO"],
            "stateProvince": ["Alaska", "Alaska", "", "", "", ""],
            "decimalLatitude": ["59.99", "60", "61.5", "90", " 6e1 ", "59"],
        }
    )
    assert select_region(table, RULES["arctic"])["decimalLatitude"].to_pylist() == ["60", "61.5", "90", " 6e1 "]


def test_arctic_excludes_missing_invalid_and_southern_latitudes():
    table = pa.table({"decimalLatitude": [None, "", "bad", "NaN", "inf", "91", "-91", "-60", "1e999"]})
    assert select_region(table, RULES["arctic"]).num_rows == 0


def test_overlap_distinguishes_containment_from_similarity():
    from dev.releases.mambo_v3.plot_overlap import overlap

    assert overlap({"a"}, {"a", "b", "c", "d"}) == (1, 0.25, 1.0)
    assert overlap({"a", "b", "c", "d"}, {"a"}) == (1, 0.25, 0.25)
    assert overlap({"a"}, {"b"}) == (0, 0.0, 0.0)


def test_other_oceania_excludes_records_not_shared_species():
    table = pa.table(
        {
            "countryCode": ["AU", "NZ", "FJ", "PG", "CK", "MG"],
            "continent": ["OCEANIA", "OCEANIA", "OCEANIA", "", "OCEANIA", "AFRICA"],
            "speciesKey": ["shared"] * 6,
        }
    )
    selected = select_region(table, RULES["oceania_excluding_australia_nz"])
    assert selected["countryCode"].to_pylist() == ["FJ", "PG", "CK"]
    assert selected["speciesKey"].to_pylist() == ["shared"] * 3
    assert select_region(table, RULES["new_zealand"])["countryCode"].to_pylist() == ["NZ"]
    assert select_region(table, RULES["madagascar"])["countryCode"].to_pylist() == ["MG"]


def test_northern_africa_intentionally_overlaps_subsaharan_transition():
    countries = ["MA", "EG", "SD", "MR", "ZA", "MG"]
    assert countries_selected("north_africa", countries) == ["MA", "EG", "SD", "MR"]
    assert countries_selected("subsaharan_africa", countries) == ["SD", "MR", "ZA", "MG"]


@pytest.mark.parametrize("preset", ["asia", "east_asia"])
def test_russian_contribution_requires_asian_continent(preset):
    table = pa.table(
        {
            "countryCode": ["RU", "RU", "RU", "RU", "JP"],
            "continent": ["ASIA", "EUROPE", "", None, ""],
            "speciesKey": ["asian_russia", "european_russia", "blank", "null", "japan"],
        }
    )
    assert select_region(table, RULES[preset])["speciesKey"].to_pylist() == ["asian_russia", "japan"]
