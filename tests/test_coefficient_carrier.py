import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nova.equilibrium.coefficient_carrier import (
    CoefficientCarrier,
    IterateRoute,
    coefficient_fixed_point_map,
    dense_newton,
    relative_exact_residual,
    select_fixed_point_map,
)


@pytest.fixture
def lattice():
    radial = np.linspace(1.4, 3.8, 11)
    vertical = np.linspace(-1.2, 1.2, 13)
    rr, zz = np.meshgrid(radial, vertical, indexing="xy")
    coordinate = np.c_[rr.ravel(), zz.ravel()]
    return coordinate, CoefficientCarrier.from_coordinates(
        coordinate, radial_knots=6, vertical_knots=6
    )


def smooth_field(coordinate):
    radial = coordinate[:, 0]
    vertical = coordinate[:, 1]
    return 0.3 + radial - 0.4 * vertical + 0.2 * radial * vertical


def test_coarse_knot_values_reuse_the_tensor_spline_evaluator(lattice):
    coordinate, carrier = lattice
    field = jnp.asarray(smooth_field(coordinate))
    represented = carrier.expand(carrier.project(field))

    np.testing.assert_allclose(represented, field, rtol=3e-13, atol=3e-13)
    assert carrier.coefficient_shape == (6, 6)
    assert carrier.expansion.shape == (len(coordinate), 36)


def test_exact_value_route_remains_explicitly_selectable_at_the_call_site(lattice):
    coordinate, carrier = lattice
    state = jnp.asarray(smooth_field(coordinate))

    def exact_map(value):
        return 0.75 * value + 0.25

    exact_route = select_fixed_point_map(IterateRoute.EXACT_VALUES, exact_map)
    coefficient_route = select_fixed_point_map(
        IterateRoute.COEFFICIENT_CARRIER, exact_map, carrier=carrier
    )

    np.testing.assert_allclose(exact_route(state), exact_map(state), rtol=0, atol=0)
    coefficient_state = carrier.project(state)
    np.testing.assert_allclose(
        coefficient_route(coefficient_state),
        carrier.project(exact_map(carrier.expand(coefficient_state))),
        rtol=2e-13,
        atol=2e-13,
    )


def test_coefficient_map_exposes_exact_output_for_physics_reads(lattice):
    coordinate, carrier = lattice
    target = jnp.asarray(smooth_field(coordinate))

    def exact_map(value):
        return target + 0.2 * (value - target)

    mapped, exact_output = coefficient_fixed_point_map(exact_map, carrier)
    coefficients = carrier.project(target + 0.03)
    output = exact_output(coefficients)

    np.testing.assert_allclose(mapped(coefficients), carrier.project(output))
    assert relative_exact_residual(output, carrier.expand(coefficients)) > 0.0
    assert jax.jit(mapped)(coefficients).shape == (carrier.coefficient_count,)


def test_dense_newton_admits_steps_using_the_exact_output(lattice):
    coordinate, carrier = lattice
    root = jnp.asarray(smooth_field(coordinate))

    def exact_map(value):
        displacement = value - root
        return root + 0.15 * displacement + 0.01 * displacement**2

    initial = carrier.project(root + 0.08)
    result = dense_newton(exact_map, carrier, initial, steps=4)

    assert result.admitted_advances >= 2
    assert result.newton_step_equivalents > 0.0
    assert float(result.exact_residual) < 1e-12
    assert np.all(np.diff(np.asarray(result.trace)) < 0.0)
    np.testing.assert_allclose(result.exact_state, root, rtol=2e-12, atol=2e-12)


def test_measurement_receipt_carries_each_required_axis():
    receipt = json.loads(
        Path("docs/figures/coefficient-space-newton/carrier-arms.json").read_text(
            encoding="utf-8"
        )
    )

    assert receipt["exact_value_route_retained"]["call_site_selection"] is True
    assert set(receipt["arms"]) == {"A", "C", "D"}
    for arm in receipt["arms"].values():
        assert arm["terminal_exact_field_residual"] >= 0.0
        assert arm["admitted_advance_count"] >= 0
        assert arm["newton_step_equivalents"] >= 0.0
        assert arm["terminal_topology"]["class"] in {"limited", "diverted"}
        assert arm["peak_device_memory_bytes"] is not None
        assert set(arm["accuracy_against_banked_root"]) == {
            "closed_flux",
            "separatrix_band",
            "scrape_off_layer",
        }
