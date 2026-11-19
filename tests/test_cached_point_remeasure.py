"""Receipt checks for the production point program served from the warm cache.

The measurement driver writes one receipt per arm per role. These checks read
the receipts one H200 job wrote — the warm loads and the cross-process identity
comparisons — and assert the load clause on them; without those receipts there
is nothing on CPU that could stand in for the card, so the checks skip rather
than pass.

Pointing ``NOVA_CACHED_POINT_REMEASURE_WARM`` at the empty-cache control
receipts of the same job is the negative control: each of those arms recorded a
miss, so the zero-miss assertion fails and the session exits nonzero.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

ARMS = ["census-2132", "census-250", "census-500", "row-132"]
COARSE_ARMS = ("census-500", "census-2132")
BYTE_BOUND = 64 * (2**30)
SECOND_BOUND = 60.0


def _receipts(variable):
    """Return the receipts one H200 job wrote, or skip when none is named."""
    directory = os.environ.get(variable)
    if not directory:
        pytest.skip(f"requires the H200 point-program receipts named by {variable}")
    receipts = [
        json.loads(path.read_text(), strict=False)
        for path in sorted(Path(directory).glob("*.json"))
    ]
    assert receipts, f"no receipts under {directory}"
    return receipts


def test_point_program_first_call_is_served_from_the_warm_cache():
    """Each arm's cached first call returns inside 60 s and 64 GB with no miss."""
    receipts = _receipts("NOVA_CACHED_POINT_REMEASURE_WARM")
    assert sorted(row["arm"] for row in receipts) == ARMS
    for row in receipts:
        assert row["role"] == "warm", row["arm"]
        assert "H200" in row["device"], row["arm"]
        assert row["cache_enabled"] is True, row["arm"]
        assert row["misses"] == 0, f"{row['arm']} compiled instead of loading"
        assert row["first_call_seconds"] <= SECOND_BOUND, row["arm"]
        assert row["first_call_peak_rss_bytes"] <= BYTE_BOUND, row["arm"]


def test_point_program_cached_rows_match_a_cold_compile():
    """The cached program returns the cold compile's bits at every compared node."""
    receipts = _receipts("NOVA_CACHED_POINT_REMEASURE_IDENTITY")
    by_arm = {row["arm"]: row for row in receipts}
    assert sorted(by_arm) == ARMS
    for arm, row in by_arm.items():
        assert row["identity"] == "bit-identical", arm
        assert row["node_coordinates_identical"], arm
        assert row["values_bitwise_identical"], arm
        assert row["derivative_entries_bitwise_identical"], arm
        assert row["rows_digest_identical"], arm
    assert by_arm["census-250"]["nodes_compared"] == by_arm["census-250"]["arm_nodes"]
    for coarse in COARSE_ARMS:
        assert by_arm[coarse]["nodes_compared"] == 64, coarse


def test_control_receipts_record_the_miss_they_are_supposed_to_record():
    """The empty-cache control's own sockets are misses with the budget at zero.

    This check reads the control receipts and asserts the condition the
    negative control of the load check depends on, so a control that silently
    loaded from somewhere would be caught here instead of making the red run
    look like a card fault.
    """
    receipts = _receipts("NOVA_CACHED_POINT_REMEASURE_CONTROL")
    assert sorted(row["arm"] for row in receipts) == ARMS
    for row in receipts:
        assert row["role"] == "control", row["arm"]
        assert row["misses"] >= 1, f"{row['arm']} recorded no miss"
        assert row["miss_budget"] == 0, row["arm"]