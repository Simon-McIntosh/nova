"""Contracts for the discrete labelled-state convention evaluation."""

from __future__ import annotations

import json
from math import tau
from pathlib import Path

import numpy as np
import pytest

from benchmarks import diiid_root_convention_sweep as sweep


def test_variant_set_exhausts_convention_digits_and_required_axes() -> None:
    members = sweep.variants()

    assert [item.source_cocos for item in members] == [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
    ]
    assert {item.poloidal_flux_sign for item in members} == {-1, 1}
    assert {item.toroidal_current_sign for item in members} == {-1, 1}
    assert {item.derivative_sign for item in members} == {-1, 1}
    assert {item.raw_flux_interpretation for item in members} == {
        "per-radian",
        "total",
    }


def test_pinned_member_reproduces_declared_transform() -> None:
    pinned = next(
        item for item in sweep.variants() if item.source_cocos == sweep.CORPUS_COCOS
    )

    assert pinned.source_cocos == 5
    assert pinned.psi_to_nova == pytest.approx(-tau)
    assert pinned.ip_to_nova == pytest.approx(1.0)
    assert pinned.derivative_to_nova == pytest.approx(-1.0 / tau)
    assert pinned.raw_flux_interpretation == "per-radian"


def test_preregistration_is_fail_closed_and_precedes_scoring(tmp_path: Path) -> None:
    path = sweep.write_preregistration(tmp_path)
    declaration = json.loads(path.read_text())

    assert declaration["no_root_search"] is True
    assert declaration["coefficients_fitted"] == 0
    assert len(declaration["variant_set"]["members"]) == 16
    changed = declaration | {"coefficients_fitted": 1}
    path.write_text(json.dumps(changed))
    with pytest.raises(RuntimeError, match="differs"):
        sweep.write_preregistration(tmp_path)


def _summary(identifier: str, fixed: float, free: float) -> dict:
    return {
        "variant": {"identifier": identifier},
        "landed_controls": {
            "at_candidate_fixed_boundary_level": (
                free <= sweep.FIXED_LEVEL_RATIO * fixed
            ),
            "free_to_fixed_median_ratio": free / fixed,
            "free_boundary_fractional_rms": {"median": free},
        },
    }


def test_verdict_distinguishes_convention_artefact_from_data_limitation() -> None:
    limitation = sweep.convention_verdict(
        [
            _summary("source-cocos-5", 0.04, 0.21),
            _summary("source-cocos-17", 8.8, 7.8),
        ]
    )
    artefact = sweep.convention_verdict(
        [
            _summary("source-cocos-5", 0.04, 0.21),
            _summary("source-cocos-17", 0.04, 0.041),
        ]
    )

    assert limitation["data_limitation_survives_convention_sweep"] is True
    assert limitation["any_variant_reaches_landed_fixed_boundary_level"] is False
    assert limitation["candidate_relative_admitted_variants"] == ["source-cocos-17"]
    assert artefact["convention_artefact"] is True
    assert artefact["admitted_variants"] == ["source-cocos-17"]


def test_linear_evaluation_uses_declared_label_and_current_factors() -> None:
    class Operator:
        def solve(self, source: np.ndarray, border: np.ndarray) -> np.ndarray:
            return np.asarray(border) + 0.01 * np.asarray(source)

    profile = {
        "raw_flux_rz": np.arange(9, dtype=float).reshape(3, 3),
        "pinned_p_prime": np.ones((3, 3)),
        "pinned_ff_prime": np.ones((3, 3)),
        "active": np.ones((3, 3), dtype=bool),
    }
    pinned = next(item for item in sweep.variants() if item.source_cocos == 5)
    measured = sweep.evaluate_variant(
        pinned,
        profile,
        np.asarray([1.0, 1.5, 2.0]),
        Operator(),
        np.zeros((3, 3)),
    )

    assert np.isfinite(measured["fixed_boundary_fractional_rms"])
    assert np.isfinite(measured["free_boundary_fractional_rms"])
    assert measured["free_to_fixed_ratio"] > 1.0


def test_generated_receipt_reproduces_controls_and_answers_question() -> None:
    path = sweep.DEFAULT_OUTPUT / sweep.RECEIPT_NAME
    receipt = json.loads(path.read_text())

    assert receipt["selection"]["frames"] == 20
    assert receipt["selection"]["landed_control_frames"] == 5
    assert receipt["selection"]["screened_additional_frames"] == 15
    assert receipt["selection"]["all_additional_absent_from_polarity_population"]
    assert receipt["control"]["reproduced"] is True
    assert receipt["control"]["measured_free_boundary_median"] == pytest.approx(
        sweep.LANDED_FREE_MEDIAN, abs=sweep.CONTROL_ABSOLUTE_TOLERANCE
    )
    assert receipt["control"]["measured_fixed_boundary_median"] == pytest.approx(
        sweep.LANDED_FIXED_MEDIAN, abs=sweep.CONTROL_ABSOLUTE_TOLERANCE
    )
    assert len(receipt["variants"]) == 16
    assert receipt["no_root_search"] is True
    assert receipt["coefficients_fitted"] == 0
    assert isinstance(
        receipt["verdict"]["any_variant_reaches_landed_fixed_boundary_level"], bool
    )
    assert (sweep.DEFAULT_OUTPUT / sweep.FIGURE_NAME).is_file()
