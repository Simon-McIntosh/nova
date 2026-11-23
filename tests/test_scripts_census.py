"""Hold the benchmark and script census to the contract its record claims.

The census decides which drivers may be deleted. These tests pin the properties
the record depends on: the population is really enumerated, the verdict counts
reconcile with it, every delete names citations that resolve, a bare sibling
import and a plan-named driver directory each keep their files, a container
directory owns nothing, the census keeps no driver through its own machinery,
and a forced delete is refused for its intended reason alone.
"""

from __future__ import annotations

import json
import subprocess

import pytest

from benchmarks import scripts_census as census

pytestmark = pytest.mark.slow

RECORD = census.RECORD_SELF + ".json"


def owning_sprints():
    """The sprint window this record declares, which the census takes as input."""
    return tuple(json.loads((census.ROOT / RECORD).read_text())["owning_sprints"])


@pytest.fixture(scope="module")
def payload():
    return census.census(owning=owning_sprints())


def test_population_holds_the_positive_control():
    files = census.population(census.tracked())
    assert census.POSITIVE in files


def test_counts_reconcile_with_the_measured_population(payload):
    counts = payload["counts"]
    summed = counts["keep"] + counts["delete"] + counts["merge"]
    assert summed == counts["total"]
    assert counts["total"] == len(census.population(census.tracked()))
    assert payload["reconciles"] is True


def test_population_is_the_tracked_python_files_of_both_trees():
    done = subprocess.run(
        ["git", "-C", str(census.ROOT), "ls-files"],
        capture_output=True,
        text=True,
        check=True,
    )
    tracked = [
        path
        for path in done.stdout.splitlines()
        if path.endswith(".py") and path.split("/", 1)[0] in census.TREES
    ]
    assert census.population(census.tracked()) == sorted(tracked)


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


def test_sibling_import_keeps_the_neighbour(payload):
    by_path = {row["path"]: row for row in payload["verdicts"]}
    row = by_path["scripts/prototypes/stored_state_refusals/panels.py"]
    assert row["verdict"] == "keep", row["reason"]
    kept_by = [
        reason for reason in row["reason"] if reason.startswith("imported_by_kept:")
    ]
    assert kept_by, row["reason"]


def test_plan_named_directory_keeps_every_file_under_it(payload):
    by_path = {row["path"]: row for row in payload["verdicts"]}
    row = by_path["scripts/iron_linear_magnetisation/gate.py"]
    assert row["verdict"] == "keep", row["reason"]
    assert any(reason.startswith("plan_owned:") for reason in row["reason"])


def test_a_container_directory_confers_no_ownership(payload):
    population = census.population(census.tracked())
    assert "scripts/prototypes" in census.CONTAINERS
    assert "scripts/prototypes" not in census.owning_directories(population)
    by_path = {row["path"]: row for row in payload["verdicts"]}
    row = by_path["scripts/prototypes/bank_batch_width/measure.py"]
    assert row["verdict"] == "delete", row["reason"]
    assert row["reason"] == []


def test_a_driver_directory_conveys_ownership(payload):
    population = census.population(census.tracked())
    assert "scripts/prototypes/read_route" in census.owning_directories(population)
    by_path = {row["path"]: row for row in payload["verdicts"]}
    held = [
        row
        for path, row in by_path.items()
        if path.startswith("scripts/prototypes/read_route/")
    ]
    assert held
    assert all(
        any(reason.startswith("plan_owned:") for reason in row["reason"])
        for row in held
    )


def test_the_census_keeps_no_driver_through_its_own_machinery(payload):
    self_references = [
        (row["path"], reason)
        for row in payload["verdicts"]
        for reason in row.get("reason") or []
        if "test_scripts_census.py" in reason
    ]
    assert self_references == []


def test_forced_delete_of_a_live_test_driver_is_refused():
    with pytest.raises(census.Refused) as raised:
        census.census(census.POSITIVE, owning_sprints())
    assert census.POSITIVE in str(raised.value)


def test_forced_delete_of_a_driver_a_kept_file_imports_is_refused():
    forced = "scripts/prototypes/stored_state_refusals/panels.py"
    with pytest.raises(census.Refused) as raised:
        census.census(forced, owning_sprints())
    message = str(raised.value)
    assert forced in message
    assert "imported_by_kept:" in message
    assert "live_runner" not in message
    assert "imported_by_core" not in message


def test_forced_delete_of_a_plan_owned_directory_file_is_refused():
    forced = "scripts/iron_linear_magnetisation/gate.py"
    with pytest.raises(census.Refused) as raised:
        census.census(forced, owning_sprints())
    message = str(raised.value)
    assert forced in message
    assert "plan_owned:" in message
    assert "live_runner" not in message
