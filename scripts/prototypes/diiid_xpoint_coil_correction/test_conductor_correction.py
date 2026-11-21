"""Checks the correction arithmetic and the rules the summarizer applies to it.

The arms this node produced were read by three pieces of logic that a solve
cannot re-check cheaply: the subtraction that turns a fitted error into an
applied current, the fraction each conductor's error is of its own measured
current, and the conditions under which a fall in distance may be read as a
share. Each has a way of going wrong that looks like a result rather than an
error, so each is asserted against a case whose answer is known by hand.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import conductor_correction
import summarize
import verify


def test_applied_currents_removes_the_fitted_error():
    names = ["F1A", "F2A"]
    measured = [-177.235, -168.819]
    applied = conductor_correction.applied_currents(
        names, measured, {"F1A": -54.058, "F2A": -52.0}
    )
    assert applied == pytest.approx([-123.177, -116.819])


def test_a_conductor_the_fit_does_not_name_keeps_its_measured_current():
    applied = conductor_correction.applied_currents(
        ["F1A", "ECOILA"], [10.0, 20.0], {"F1A": 4.0}
    )
    assert applied == [6.0, 20.0]


def test_a_short_current_vector_is_refused_rather_than_truncated():
    try:
        conductor_correction.applied_currents(["F1A", "F2A"], [1.0], {})
    except ValueError:
        return
    raise AssertionError("a missing current was silently dropped")


def test_the_fraction_is_signed_and_zero_where_no_current_flows():
    fractions = conductor_correction.relative_correction([-100.0, 0.0], [50.0, 7.0])
    assert fractions == [-0.5, 0.0]


def test_the_fraction_is_read_from_the_fit_not_from_the_applied_arm(tmp_path):
    """The base arm's own correction is zero by construction, so reading the
    fraction from it reports every conductor as within tolerance."""

    case = "frame-137-diverted"
    base = tmp_path / "base" / case
    base.mkdir(parents=True)
    (base / "current-inputs.json").write_text(
        json.dumps(
            {
                "names": ["F1A", "ECOILA"],
                "measured_a_turn": [-177235.04, -12993.14],
                "correction_a_turn": [0.0, 0.0],
                "applied_a_turn": [-177235.04, -12993.14],
            }
        )
    )
    fits = tmp_path / "fits" / case
    fits.mkdir(parents=True)
    (fits / "conductor-fit-137.json").write_text(
        json.dumps({"equivalent_error_a_turn": [54058.0, -321330.0]})
    )
    rows, fit = summarize.coil_rows(tmp_path, case, 137)
    assert fit is not None
    assert rows[0]["relative_correction"] > 0.05
    assert rows[0]["implausible"] is True
    assert rows[1]["implausible"] is True


def _row(converged=True, support=339, distance=2.360):
    return {
        "converged": 1 if converged else 0,
        "termination": "CONVERGED",
        "support_cells": support,
        "distance_cm": distance,
    }


def test_a_fall_is_read_when_the_terminal_state_is_the_same_plasma():
    row = {"base": _row(distance=2.360), "corrected": _row(distance=2.0)}
    share, reason = summarize.read_only(row)
    assert reason is None
    assert abs(share - 0.1525) < 1e-3


def test_a_fall_is_not_read_when_the_corrected_arm_did_not_converge():
    row = {
        "base": _row(),
        "corrected": dict(_row(converged=False), termination="ACTIVE_SET_STEP_REFUSED"),
    }
    share, reason = summarize.read_only(row)
    assert share is None
    assert "ACTIVE_SET_STEP_REFUSED" in reason


def test_the_corpus_column_shift_is_the_negated_error_in_kiloamps():
    """A column read as kiloamps at unit turn multiplier moves by error/1000."""

    assert conductor_correction.column_shift(54058.0, 1.0) == pytest.approx(-54.058)


def test_the_shifted_column_reaches_the_error_the_fit_asked_for():
    """F1A on 137D: measured -177.235 kA.turn less a +54.058 error reads -231.293."""

    source_time = [0.0, 1000.0]
    measured_column = [-177.235, -177.235]
    shift = conductor_correction.column_shift(54058.0, 1.0)
    shifted = [value + shift for value in measured_column]
    applied = conductor_correction.current_at(source_time, shifted, 500.0, 1.0)
    assert applied / 1000.0 == pytest.approx(-231.293, abs=1e-6)


def test_the_solved_current_check_reddens_when_the_correction_did_not_land():
    """Pairing the measured arm with itself is the negative control: it fails."""

    measured = {"F1A": -177235.04, "F2A": -168819.0}
    correction = {"F1A": 54058.0, "F2A": 52140.0}
    _rows, mismatched = verify.compare(measured, measured, correction)
    assert mismatched == ["F1A", "F2A"]


def test_the_solved_current_check_passes_when_the_correction_landed():
    measured = {"F1A": -177235.04, "F2A": -168819.0}
    correction = {"F1A": 54058.0, "F2A": 52140.0}
    fitted = {name: value - correction[name] for name, value in measured.items()}
    _rows, mismatched = verify.compare(measured, fitted, correction)
    assert mismatched == []


def test_a_fall_is_not_read_when_the_support_moved():
    row = {"base": _row(support=339), "corrected": _row(support=290)}
    share, reason = summarize.read_only(row)
    assert share is None
    assert "support moved" in reason
