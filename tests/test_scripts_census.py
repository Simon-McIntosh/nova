"""Hold the benchmark and script census to the contract its record claims.

The census decides which drivers may be deleted. These tests pin the three
properties the record depends on: the population is really enumerated, the
verdict counts reconcile with it, every delete names citations that resolve, and
a driver a live test imports cannot be forced to delete.
"""

from __future__ import annotations

import pytest

from benchmarks import scripts_census as census

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def payload():
    return census.census()


def test_population_holds_the_positive_control():
    files = census.population(census.tracked())
    assert census.POSITIVE in files


def test_counts_reconcile_with_the_measured_population(payload):
    counts = payload["counts"]
    summed = counts["keep"] + counts["delete"] + counts["merge"]
    assert summed == counts["total"]
    assert counts["total"] == len(census.population(census.tracked()))
    assert payload["reconciles"] is True


def test_every_delete_citation_resolves(payload):
    for row in payload["verdicts"]:
        if row["verdict"] != "delete":
            continue
        for cite in row.get("evidence", []):
            assert cite["revision"] is not None, row["path"]
            assert cite["resolves"] is True, row["path"]


def test_every_merge_folds_into_a_kept_driver(payload):
    by_path = {row["path"]: row for row in payload["verdicts"]}
    for row in payload["verdicts"]:
        if row["verdict"] != "merge":
            continue
        assert by_path[row["merge_into"]]["verdict"] == "keep", row["path"]


def test_forced_delete_of_a_live_test_driver_is_refused(payload):
    with pytest.raises(census.Refused) as raised:
        census.census(census.POSITIVE)
    assert census.POSITIVE in str(raised.value)