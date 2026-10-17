"""CPU checks on the agreement guard, which needs no device.

The guard exists to fail when the low-order outer rule disagrees with the
reference. It did not, because the ladder it reduces over carried a row at the
reference degree itself: that row compares the reference with the reference and
reads zero however wrong the cheaper degrees are, so the minimum over the
ladder was always zero and the guard always passed. These checks pin both
halves of the repair -- the ladder excludes the reference degree, and the
decision rule returns a failure on the disagreements the run actually measured.

The recorded values are the ladder rows of bar-agreement.json from the H200 bar
gate, which measured 0.211, 0.0483, 0.00405 and 0.00658 at degrees 8, 16, 32
and 64 against a 1e-12 tolerance.
"""

import pytest

from scripts.prototypes.saddle_expansion.speed_gate import (
    AGREEMENT_TOLERANCE,
    DEGREE_LADDER,
    REFERENCE_OUTER_DEGREE,
    agreement_failures,
    agreement_passed,
)

RECORDED_LADDER = {
    "8": {"max_abs_difference": 0.2110304721853531},
    "16": {"max_abs_difference": 0.04833218463634381},
    "32": {"max_abs_difference": 0.004051944956813713},
    "64": {"max_abs_difference": 0.006582352677924064},
}


def test_ladder_excludes_the_reference_degree():
    """A ladder carrying the reference degree can never fail."""
    assert REFERENCE_OUTER_DEGREE not in DEGREE_LADDER


def test_ladder_is_not_empty():
    """An empty ladder would pass vacuously."""
    assert DEGREE_LADDER


def test_recorded_disagreement_fails_the_guard():
    """The measured ladder is four orders above tolerance at its best degree."""
    assert agreement_passed(RECORDED_LADDER) is False
    assert [degree for degree, _ in agreement_failures(RECORDED_LADDER)] == [
        8,
        16,
        32,
        64,
    ]


def test_a_converged_ladder_passes():
    """The check is not vacuous: a ladder at tolerance passes."""
    converged = {str(degree): {"max_abs_difference": 1e-13} for degree in DEGREE_LADDER}
    assert agreement_passed(converged) is True


def test_the_reference_row_would_have_passed_the_broken_guard():
    """The defect this repair closes, stated as a check on the old behaviour."""
    broken = dict(RECORDED_LADDER)
    broken[str(REFERENCE_OUTER_DEGREE)] = {"max_abs_difference": 0.0}
    passed_on_minimum = (
        min(row["max_abs_difference"] for row in broken.values()) <= AGREEMENT_TOLERANCE
    )
    assert passed_on_minimum is True  # the zero reference row alone satisfied it
    assert agreement_passed(broken) is False  # the repaired guard still refuses


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
