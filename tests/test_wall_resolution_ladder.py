"""Receipt and geometry checks for the isolated wall-count measurement."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from benchmarks.wall_resolution_ladder import (
    SCHEMA,
    _convergence,
    sample_authored_wall,
    validate_receipt,
)


def _authored_wall() -> np.ndarray:
    angle = np.linspace(0.0, 2.0 * np.pi, 37)
    wall = np.column_stack((1.0 + 0.4 * np.cos(angle), 0.8 * np.sin(angle)))
    wall[-1] = wall[0]
    return wall


def _receipt() -> dict:
    digests = {
        "plasma_carrier_sha256": "a" * 64,
        "source_policy_sha256": "b" * 64,
        "conductor_currents_sha256": "c" * 64,
        "seed_field_sha256": "d" * 64,
        "plasma_seed_prefix_sha256": "e" * 64,
    }
    rows = []
    for count in (37, 36, 72, 144, 288):
        rows.append(
            {
                "wall_target_count": count,
                "boundary_flux_wb": 0.1,
                "contact_coordinate_m": [0.5, -1.0],
                "achieved_class": "limited",
                "class_margin": -0.2,
                "residual_trajectory": [1.0, 0.5],
                "residual_trajectory_sha256": "f" * 64,
                "fixed_carrier_digests": digests,
            }
        )
    return {
        "schema": SCHEMA,
        "control_reproduction": {"passes": True},
        "rows": rows,
    }


@pytest.mark.parametrize("samples,count", [(1, 36), (2, 72), (4, 144), (8, 288)])
def test_each_authored_segment_realises_the_declared_wall_count(samples, count):
    sampled = sample_authored_wall(_authored_wall(), samples)
    assert sampled.shape == (count, 2)
    assert len(np.unique(sampled, axis=0)) == count


def test_the_count_clouds_are_nested_in_the_finest_cloud():
    wall = _authored_wall()
    fine = sample_authored_wall(wall, 8)
    for samples in (1, 2, 4):
        coarse = sample_authored_wall(wall, samples)
        np.testing.assert_array_equal(coarse, fine[:: 8 // samples])


def test_the_unique_baseline_removes_only_the_closure_duplicate():
    wall = _authored_wall()
    np.testing.assert_array_equal(sample_authored_wall(wall, 1), wall[:-1])


def test_receipt_validation_requires_fixed_carrier_identity():
    payload = _receipt()
    payload["rows"][3]["fixed_carrier_digests"] = {
        **payload["rows"][3]["fixed_carrier_digests"],
        "seed_field_sha256": "0" * 64,
    }
    with pytest.raises(ValueError, match="fixed-carrier identity changed"):
        validate_receipt(payload)


def test_receipt_validation_requires_the_complete_count_sequence():
    payload = _receipt()
    payload["rows"][2]["wall_target_count"] = 71
    with pytest.raises(ValueError, match="unexpected wall count sequence"):
        validate_receipt(payload)


def test_receipt_validation_accepts_the_isolated_ladder():
    result = validate_receipt(copy.deepcopy(_receipt()))
    assert result == {
        "valid": True,
        "row_count": 5,
        "wall_target_counts": [37, 36, 72, 144, 288],
    }


def test_convergence_reports_unresolved_limiter_motion():
    rows = [
        {
            "wall_target_count": count,
            "boundary_flux_wb": boundary_flux,
            "contact_coordinate_m": [0.5, contact_z],
            "achieved_class": "limited",
            "spacing": {"maximum_m": maximum_spacing},
        }
        for count, boundary_flux, contact_z, maximum_spacing in (
            (36, -0.03, 1.8, 0.5),
            (72, -0.025, 1.78, 0.3),
            (144, -0.026, 1.77, 0.2),
            (288, -0.0261, 1.769, 0.15),
        )
    ]

    result = _convergence(rows, plasma_pitch=0.125)

    assert result["first_plasma_equivalent_wall_count"] is None
    assert result["limiter_operand_stop_count"] is None
    assert rows[-1]["maximum_spacing_in_plasma_pitches"] == pytest.approx(1.2)
    assert "does not stop moving by 288 targets" in result["statement"]
