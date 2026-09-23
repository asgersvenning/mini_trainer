"""Behavioral contracts for deliberately overlapping deployment regions."""

import tomllib

import pytest

from dev.releases.mambo_v3.audit import HERE
from dev.releases.mambo_v3.build_presets import ordered_membership, select_region

pa = pytest.importorskip("pyarrow")
RULES = tomllib.loads((HERE / "preset-definitions.toml").read_text())["presets"]


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
    with pytest.raises(ValueError, match="requires countries or continents"):
        select_region(table, {})
