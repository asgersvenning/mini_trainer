"""Regional selection must preserve row counts and the source geography field."""

import pytest

from dev.releases.mambo_v3.reconstruct_presets import membership, region_counts

pa = pytest.importorskip("pyarrow")


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
