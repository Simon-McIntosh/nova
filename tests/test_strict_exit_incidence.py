"""Regression checks for the allocation-free strict-exit batch proof."""

from __future__ import annotations

import pytest

from benchmarks import strict_exit_incidence as incidence


def test_batched_self_check_requires_explicit_batched_mode():
    with pytest.raises(ValueError, match="--self-check requires --batched"):
        incidence._validate_arguments(batched=False, self_check=True)


def test_select_member_keeps_scalar_receipt_leaves_unindexed():
    selected = incidence._select_member(
        {
            "per_member": incidence.jnp.asarray([3, 5]),
            "scalar": incidence.jnp.asarray(7),
        },
        1,
    )

    assert int(selected["per_member"]) == 5
    assert int(selected["scalar"]) == 7


def test_batch_member_result_selects_the_sequential_fallback_member():
    results = ("first", "second")

    assert incidence._batch_member_result(results, 1, stacked=False) == "second"
