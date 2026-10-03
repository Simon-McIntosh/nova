"""Pin the recovery-ladder census to every entry the module defines.

The census wraps each promotion and recovery ladder entry so its runtime
executions are counted.  These tests hold the enumeration to the module: a
renamed rung must fail the census by name rather than drop out as a quiet zero.
"""

from __future__ import annotations

import pytest

from benchmarks import recovery_ladder_census as census
from nova.equilibrium import fixed_point


def test_module_under_test_is_the_imported_copy():
    print("fixed_point.__file__ =", fixed_point.__file__)
    assert fixed_point.__file__


def test_enumeration_finds_exactly_the_ladder_entries():
    found = census.enumerated_entries(fixed_point)
    assert len(found) == 5
    assert set(found) == set(census.LADDER_ENTRIES)


def test_counted_entries_match_the_module():
    assert set(census.counted_entries(fixed_point)) == set(census.LADDER_ENTRIES)


def test_renamed_entry_fails_naming_it(monkeypatch):
    monkeypatch.delattr(fixed_point, "_rebuilt_model_promotion")
    with pytest.raises(RuntimeError) as raised:
        census.counted_entries(fixed_point)
    assert "_rebuilt_model_promotion" in str(raised.value)


def test_instrumented_refuses_an_absent_counted_path(monkeypatch):
    monkeypatch.delattr(fixed_point, "_steepest_descent_promotion")
    with pytest.raises(RuntimeError) as raised:
        with census.instrumented(fixed_point):
            pass
    assert "_steepest_descent_promotion" in str(raised.value)
