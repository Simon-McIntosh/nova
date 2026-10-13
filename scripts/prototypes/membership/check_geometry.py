"""Independent geometric and recorded numerical controls for saddle membership."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from nova.equilibrium import topology
from scripts.prototypes.membership.hybrid import neighbour_mask

assert jax.config.jax_enable_x64 is True


def test_normal_form_cancels_cubic_flux():
    hessian = jnp.diag(jnp.asarray((-2.0, 3.0)))
    third = jnp.asarray((((1.8, 0.4), (0.4, -0.2)), ((0.4, -0.2), (-0.2, 0.6))))
    form = topology.saddle_normal_form(jnp.zeros(2), hessian, third, 1e-12)
    assert form.valid
    residual = jnp.einsum("ai,ij,aj->a", form.direction, hessian, form.curvature)
    residual += (
        jnp.einsum(
            "ijk,ai,aj,ak->a", third, form.direction, form.direction, form.direction
        )
        / 6
    )
    np.testing.assert_allclose(residual, 0, atol=1e-15)
    assert float(jnp.max(jnp.abs(form.curvature))) > 0.01
    straight = topology.saddle_normal_form(
        jnp.zeros(2), hessian, jnp.zeros_like(third), 1e-12
    )
    np.testing.assert_array_equal(straight.curvature, 0)
    np.testing.assert_array_equal(straight.cubic, 0)


def test_neighbours_are_edge_connected_not_a_distance_disc():
    from types import SimpleNamespace

    # A corner-only contact must not enter the local normal-form patch.
    origins = np.asarray(((0, 0), (1, 0), (0, 1), (1, 1)), float)
    square = np.asarray(((0, 0), (1, 0), (1, 1), (0, 1)), float)
    geometry = SimpleNamespace(
        vertices=jnp.asarray(origins[:, None] + square),
        vertex_count=jnp.full(4, 4),
        pitch=jnp.ones(4),
        neighbour=jnp.asarray(
            ((-1, 1, 2, -1), (-1, -1, 3, 0), (0, 3, -1, -1), (1, -1, -1, 2))
        ),
    )
    mask = neighbour_mask(geometry, jnp.asarray((0.5, 0.5)))
    np.testing.assert_array_equal(mask, [True, True, True, False])


def _receipt():
    path = Path(os.environ["MEMBERSHIP_RECEIPT"])
    value = json.loads(path.read_text())
    assert value["complete"], path
    return value, {row["arm"]: row for row in value["rows"]}


def test_closed_form_arm_reproduces():
    receipt, rows = _receipt()
    assert receipt["closed_form_delta"] < 1e-9
    assert rows["closed_form"]["valid"]
    assert receipt["reference_uncertainty"] < 1e-7


def test_zero_third_reproduces_straight_saddle_error():
    receipt, rows = _receipt()
    assert receipt["kind"] == "diverted" and receipt["cells"] == 2000
    measured = rows["booked_zero_third"]["saddle_error"]
    expected = rows["recorded_straight"]["saddle_error"]
    print(f"ZERO_THIRD_SADDLE measured={measured} straight={expected}", flush=True)
    assert rows["booked_zero_third"]["valid"]
    assert abs(expected - 0.005919276) < 1e-8
    # Booked moments and cached stationary points differ from the closed form.
    assert abs(measured - expected) < 3e-4


def test_curved_saddle_beats_both_references():
    receipt, rows = _receipt()
    arm = os.environ.get("MEMBERSHIP_ARM", "booked_curved")
    measured = rows[arm]
    target = min(
        rows["booked_bell"]["saddle_error"],
        rows["recorded_straight"]["saddle_error"],
    )
    print(
        f"MEMBERSHIP_TARGET arm={arm} error={measured['saddle_error']} target={target}"
    )
    assert measured["valid"]
    assert measured["saddle_error"] < target


def test_curvature_contributes_to_saddle_improvement():
    _, rows = _receipt()
    arm = os.environ.get("MEMBERSHIP_ARM", "booked_curved")
    measured = rows[arm]
    straight = rows["booked_zero_third"]
    print(
        f"CURVATURE_CONTRIBUTION arm={arm} error={measured['saddle_error']} "
        f"zero_third={straight['saddle_error']}"
    )
    assert measured["valid"] and straight["valid"]
    assert measured["saddle_error"] < straight["saddle_error"]


def test_curve_crossing_patch_uses_intersection_distance():
    from scripts.prototypes.membership.hybrid import crossed_within_radius

    form = topology.saddle_normal_form(
        jnp.zeros(2),
        jnp.diag(jnp.asarray((-2.0, 2.0))),
        jnp.zeros((2, 2, 2)),
        1e-12,
    )
    square = jnp.asarray(((0.4, 0.4), (0.6, 0.4), (0.6, 0.6), (0.4, 0.6)))
    check = jax.jit(crossed_within_radius)
    assert check(square, 4, form, 0.6)
    assert not check(square, 4, form, 0.55)
    assert not check(square + jnp.asarray((0.0, -0.6)), 4, form, 0.6)


def test_outside_fragments_are_byte_identical():
    _, rows = _receipt()
    arms = [
        row for name, row in rows.items() if name.endswith(("_preserved", "_radius"))
    ]
    assert len(arms) >= 2
    for row in arms:
        assert row["outside_byte_identical"]
        assert row["outside_area_byte_identical"]
        assert row["outside_edge_byte_identical"]
        assert row["outside_selection_changes"] == 0


def test_outside_mismatch_is_local_fragment_geometry():
    receipt, _ = _receipt()
    diagnosis = receipt["outside_diagnosis"]
    assert not diagnosis["immediate_patch"]
    assert diagnosis["bell_normal_form"]
    assert (
        abs(diagnosis["conic_fraction_at_bell_level"] - diagnosis["conic_fraction"])
        < 1e-13
    )
    assert abs(diagnosis["conic_fraction"] - diagnosis["bell_fraction"]) > 1e-5
    assert max(diagnosis["conic_area"]) > 0
    assert max(diagnosis["bell_area"]) > 0
