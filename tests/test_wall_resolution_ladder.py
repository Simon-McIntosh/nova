"""Receipt and geometry checks for the isolated wall-count measurement."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.wall_resolution_ladder import (
    SCHEMA,
    _control_batch_coordinates,
    _control_reproduction_passes,
    _convergence,
    _json_digest,
    _plasma_carrier_digest,
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
        "terminal_plasma_flux_sha256": "f" * 64,
    }
    trajectory = [1.0, 0.5]
    rows = []
    for count in (37, 36, 72, 144, 288):
        rows.append(
            {
                "wall_target_count": count,
                "boundary_flux_wb": 0.1,
                "contact_coordinate_m": [0.5, -1.0],
                "achieved_class": "limited",
                "class_margin": -0.2,
                "residual_trajectory": trajectory.copy(),
                "residual_trajectory_sha256": _json_digest(trajectory),
                "fixed_carrier_digests": digests,
            }
        )
    exact_audit = {
        name: {"array_equal": True}
        for name in ("source_to_wall", "plasma_to_wall", "prescribed_full")
    }
    return {
        "schema": SCHEMA,
        "control_reproduction": {
            "authored_coordinate_exact": True,
            "control_operator_wall_coordinate_exact": True,
            "control_seed_exact": True,
            "original_batch_contract": {
                "source_shape_37_by_2": True,
                "source_authored_order_exact": True,
                "prescribed_shape_1126_by_2": True,
                "prescribed_grid_before_wall_order_exact": True,
            },
            "direct_control_audit": exact_audit,
            "passes": True,
        },
        "isolation": {"fixed_carrier_digests": digests},
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


def test_control_batches_preserve_original_shapes_and_order():
    wall = _authored_wall()
    grid = np.column_stack(
        (
            np.linspace(0.2, 1.8, 1089),
            np.linspace(-1.5, 1.5, 1089),
        )
    )

    source, prescribed = _control_batch_coordinates(
        {"grid_coordinate": grid, "wall_coordinate": wall}, wall
    )

    assert source.shape == (37, 2)
    assert prescribed.shape == (1126, 2)
    np.testing.assert_array_equal(source, wall)
    np.testing.assert_array_equal(prescribed[:1089], grid)
    np.testing.assert_array_equal(prescribed[1089:], wall)


def test_control_batch_rejects_a_perturbed_authored_row():
    wall = _authored_wall()
    perturbed = wall.copy()
    perturbed[18, 0] += 1.0e-12
    grid = np.zeros((1089, 2))

    with pytest.raises(ValueError, match="changed authored wall order"):
        _control_batch_coordinates(
            {"grid_coordinate": grid, "wall_coordinate": wall}, perturbed
        )


def test_terminal_plasma_flux_is_part_of_carrier_identity():
    grid = SimpleNamespace(
        source_target=np.arange(4.0).reshape(2, 2),
        plasma_target=np.arange(6.0).reshape(2, 3),
        plasma_target_r=np.array([0.5, 0.6]),
        plasma_target_z=np.array([-0.1, 0.1]),
    )
    profile = SimpleNamespace(
        operator=SimpleNamespace(
            grid=grid,
            area=np.array([0.1, 0.2]),
            inside_material=np.array([False, True]),
            declared_support=np.array([True, True]),
        )
    )
    case = {"grid_coordinate": np.array([[0.5, -0.1], [0.6, 0.1]])}
    terminal = np.array([0.01, 0.02])

    original = _plasma_carrier_digest(case, profile, terminal)
    changed = _plasma_carrier_digest(case, profile, terminal + [0.0, 1.0e-12])

    assert original != changed


def test_receipt_validation_recomputes_each_residual_trajectory_digest():
    payload = _receipt()
    payload["rows"][2]["residual_trajectory"][1] = 0.4
    with pytest.raises(ValueError, match="digest does not match its values"):
        validate_receipt(payload)


def test_receipt_validation_requires_top_level_carrier_identity():
    payload = _receipt()
    payload["rows"][3]["fixed_carrier_digests"] = {
        **payload["rows"][3]["fixed_carrier_digests"],
        "seed_field_sha256": "0" * 64,
    }
    with pytest.raises(ValueError, match="differs from the top-level isolation map"):
        validate_receipt(payload)


def test_nonexact_response_row_fails_control_reproduction():
    payload = _receipt()
    payload["control_reproduction"]["direct_control_audit"]["prescribed_full"][
        "array_equal"
    ] = False
    payload["control_reproduction"]["passes"] = False

    assert not _control_reproduction_passes(payload["control_reproduction"])
    with pytest.raises(ValueError, match="37-row control did not reproduce"):
        validate_receipt(payload)


def test_validation_does_not_trust_a_control_pass_flag():
    payload = _receipt()
    payload["control_reproduction"]["direct_control_audit"]["source_to_wall"][
        "array_equal"
    ] = False

    with pytest.raises(ValueError, match="pass flag disagrees with exact audits"):
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
