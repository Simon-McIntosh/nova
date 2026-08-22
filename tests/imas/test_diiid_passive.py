"""Invariants for the DIII-D wall-following passive-loop description."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from nova.imas import diiid_passive as passive


def test_description_authors_one_polygon_per_slender_loop() -> None:
    description = passive.build_description()

    assert len(description.loops) == passive.LOOP_COUNT == 48
    assert len(description.pf_passive.loop) == 48
    assert len(description.limiter_contour_rz_m) == 82
    for record, ids_loop in zip(
        description.loops, description.pf_passive.loop, strict=True
    ):
        assert len(ids_loop.element) == 1
        assert len(ids_loop.element[0].geometry.outline.r) == len(record.outline_rz_m)
        assert ids_loop.element[0].area == pytest.approx(record.area_m2)
        assert record.aspect_ratio >= passive.MINIMUM_SLENDERNESS
        assert record.through_thickness_m == passive.NOMINAL_STRUCTURAL_THICKNESS_M


def test_wall_band_area_closes_against_banked_limiter_authority() -> None:
    description = passive.build_description()
    meshed_area = sum(loop.area_m2 for loop in description.loops)

    assert description.source_receipt["mesh"]["topology_repair"]["required"]
    assert description.area_relative_error <= passive.AREA_RELATIVE_TOLERANCE
    assert abs(meshed_area / description.wall_band_area_m2 - 1.0) <= 1.0e-4
    assert description.excluded_repair_area_m2 == pytest.approx(5.7936427518673954e-5)


def test_resistance_is_positive_and_keeps_thickness_envelopes_distinct() -> None:
    description = passive.build_description()

    for loop in description.loops:
        assert 0.0 < loop.resistance_lower_ohm <= loop.resistance_ohm
        assert loop.resistance_ohm <= loop.resistance_upper_ohm
        assert loop.thincurr_resistance_lower_ohm > loop.resistance_lower_ohm
        assert loop.thincurr_resistance_upper_ohm > loop.resistance_upper_ohm
    assert passive.STRUCTURAL_THICKNESS_RANGE_M == (0.025, 0.038)
    assert passive.THINCURR_EFFECTIVE_THICKNESS_RANGE_M == (0.010, 0.025)


def test_slender_self_inductance_is_exactly_reciprocal() -> None:
    description = passive.build_description()
    operator = description.self_inductance_operator_wb_per_a

    assert operator.shape == (48, 48)
    assert np.array_equal(operator, operator.T)
    assert np.isfinite(np.linalg.cond(operator))


def test_committed_receipt_carries_static_model_qualifications() -> None:
    receipt_path = Path(
        "docs/figures/coil-circuit-discovery/pf_passive_description_receipt.json"
    )
    figure_path = Path("docs/figures/coil-circuit-discovery/pf_passive_vessel_mesh.png")
    operator_path = Path(
        "docs/figures/coil-circuit-discovery/pf_passive_slender_operator.npz"
    )
    receipt = json.loads(receipt_path.read_text())

    assert receipt["mesh"]["loop_count"] == 48
    assert receipt["mesh"]["wall_band_area_relative_error"] <= 1.0e-4
    assert receipt["mesh"]["aspect_ratio"]["minimum"] >= 5.0
    assert receipt["scope"]["dynamic_vessel_current_model_claimed"] is False
    assert receipt["self_inductance"]["shape"] == [48, 48]
    assert receipt["self_inductance"]["maximum_asymmetry_wb_per_a"] == 0.0
    assert receipt["self_inductance"]["hex_operator_comparison"] == {
        "condition_number_2": pytest.approx(372.65522630279304),
        "diagonal_dominance_ratio": pytest.approx(0.029841680213628683),
    }
    assert "eigenmode" in receipt["self_inductance"]["eigenmode_reduction_caveat"]
    assert (
        "never silently equated"
        in receipt["material_and_resistance"]["uncertainty_statement"]
    )
    assert len(receipt["loops"]) == 48
    assert figure_path.stat().st_size > 0
    assert operator_path.stat().st_size > 0
