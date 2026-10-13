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
