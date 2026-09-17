"""Bounded exterior-field rows on the analytic fixture."""

from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from scripts.analytic_oracle_fixtures.centroid_row import (
    DEFAULT_FIELD_BOUND_T,
    DEFAULT_FIELD_SCALE_T,
    DEFAULT_STEP_LIMIT,
    centroid_constraint_pair,
    exterior_field_identity,
)
from scripts.analytic_oracle_fixtures.measure import (
    analytic_case,
    exact_state,
    gauge_free_flux_read,
    uniform_exterior_field_flux,
    uniform_exterior_field_response,
)
from tests.rotating_equilibrium_references import reference_cases


def _reactor_case():
    """Return the reactor-scale closed form the banked control rows use."""
    return reference_cases()["weak-rotation-reactor"].static_limit()


def test_uniform_field_contribution_is_zero_on_the_compensator_anchor() -> None:
    case = _reactor_case()
    axis = np.asarray(case.magnetic_axis, dtype=np.float64)
    field = np.asarray((2.0e-3, -3.0e-4), dtype=np.float64)

    on_anchor = uniform_exterior_field_flux(case, axis.reshape(1, 2), field)
    np.testing.assert_allclose(on_anchor[0], 0.0, rtol=0.0, atol=0.0)

    # the reference radius is the axis, so the column grows with R**2 outward
    radius = 7.718433652945465
    outward = uniform_exterior_field_flux(case, np.asarray(((radius, 0.0),)), field)
    np.testing.assert_allclose(
        outward[0], field[0] * np.pi * (radius**2 - float(axis[0]) ** 2), rtol=0.0
    )
    assert outward[0] > 0.0


def test_gauge_free_span_offset_ignores_an_added_flux_constant() -> None:
    case = _reactor_case()
    axis = np.asarray(case.magnetic_axis, dtype=np.float64)
    boundary = np.asarray(((7.718433652945465, -1.2320178076396662),))
    field = np.asarray((0.002387200675934618, 0.0), dtype=np.float64)
    analytic = exact_state(case, np.vstack((axis, boundary)))

    reading = gauge_free_flux_read(
        case, axis, boundary[0], float(analytic[0]), float(analytic[1]), field
    )
    np.testing.assert_allclose(reading["gauge_free_flux_offset_wb"], 0.0, atol=1e-9)

    # a constant added to both levels is a gauge choice, not a fixed-point error
    shifted = gauge_free_flux_read(
        case,
        axis,
        boundary[0],
        float(analytic[0]) + 12.5,
        float(analytic[1]) + 12.5,
        field,
    )
    np.testing.assert_allclose(
        shifted["gauge_free_flux_offset_wb"],
        reading["gauge_free_flux_offset_wb"],
        rtol=0.0,
        atol=0.0,
    )

    # the authored zero level sits on the analytic separatrix, not at the axis
    np.testing.assert_allclose(
        float(exact_state(case, np.asarray(((7.801395788446069, 0.0),)))[0]),
        0.0,
        atol=1.0e-6,
    )


def test_uniform_exterior_response_has_declared_field_components() -> None:
    case = analytic_case()
    points = np.asarray(
        ((1.1, -0.2), (1.1, 0.3), (1.6, -0.2), (1.6, 0.3)),
        dtype=np.float64,
    )
    machine = SimpleNamespace(
        node=points[:2],
        wall_node=points[2:3],
        sample_coordinates=points[3:],
    )

    response = uniform_exterior_field_response(case, machine)
    radius = points[:, 0]
    vertical_radial_derivative = (response[2:, 0] - response[:2, 0]) / 0.5
    radial_midpoint = 0.5 * (radius[2:] + radius[:2])
    radial_vertical_derivative = response[[1, 3], 1] - response[[0, 2], 1]

    np.testing.assert_allclose(
        vertical_radial_derivative / (2.0 * np.pi * radial_midpoint),
        1.0,
        rtol=0.0,
        atol=3.0e-16,
    )
    np.testing.assert_allclose(
        -radial_vertical_derivative / (2.0 * np.pi * radius[[0, 2]] * 0.5),
        1.0,
        rtol=0.0,
        atol=3.0e-16,
    )


def test_centroid_pair_maps_rows_to_bounded_field_directions() -> None:
    pair = centroid_constraint_pair(
        jnp.asarray([1.4, 0.0]),
        pitch=0.05,
    )

    assert pair.functional.components == ("centroid_r", "centroid_z")
    np.testing.assert_array_equal(pair.unknown.direction, np.eye(2))
    np.testing.assert_allclose(pair.unknown.field_bound, DEFAULT_FIELD_BOUND_T)
    assert exterior_field_identity()["centroid_r_compensator"] == (
        "uniform vertical field"
    )
    with np.testing.assert_raises_regex(
        ValueError, "exterior-field amplitude exceeds its declared finite bound"
    ):
        pair.unknown.require_within_bound(jnp.asarray([1.0e6, 0.0]))


def test_centroid_pair_step_control_caps_and_records_refusal() -> None:
    pair = centroid_constraint_pair(
        jnp.asarray([1.4, 0.0]),
        pitch=0.05,
    )
    np.testing.assert_allclose(pair.unknown.step_limit, DEFAULT_STEP_LIMIT)

    # a row residual asking for more than the cap is damped, not clipped:
    # the per-trip change lands exactly on the cap and the tangent survives
    step, refused = pair.unknown.damped_step(jnp.zeros(2), jnp.asarray([10.0, 0.0]))
    np.testing.assert_allclose(step[0], -DEFAULT_STEP_LIMIT, rtol=0.0, atol=0.0)
    assert not bool(np.asarray(refused).any())

    # a physical field past the declared bound is refused and the step holds
    over_bound = jnp.full(2, 2.0 * DEFAULT_FIELD_BOUND_T / DEFAULT_FIELD_SCALE_T)
    step, refused = pair.unknown.damped_step(over_bound, jnp.zeros(2))
    np.testing.assert_array_equal(step, np.zeros(2))
    assert bool(np.asarray(refused).all())
