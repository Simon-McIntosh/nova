"""Hold below every first knot, including negative flux on zero-start grids.

Bit identity to the capped evaluator applies for psi_norm >= grid[0], including
above grid[-1]. Flux below the normalizing axis value still carries core current.
"""

import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path
import pytest

from nova.equilibrium.solve_request import SampledFluxFunction
from nova.jax.config import configure_dtypes

configure_dtypes()


def capped_reference(profile, points):
    """Evaluate the two quintic end caps with array-owned edge increments."""
    grid, values = profile.coordinate, profile.values
    below = points < grid[0]
    edge = jnp.where(below, values[0], values[-1])
    increment = jnp.where(
        below, -profile.lower_edge_increment, profile.upper_edge_increment
    )
    distance = jnp.where(below, grid[0] - points, points - grid[-1])
    parameter = jnp.clip(distance / profile.edge_width, 0.0, 1.0)
    cube = parameter * parameter * parameter
    value_basis = 1.0 + cube * (-10.0 + parameter * (15.0 - 6.0 * parameter))
    slope_basis = parameter + cube * (-6.0 + parameter * (8.0 - 3.0 * parameter))
    return jnp.where(
        (points >= grid[0]) & (points <= grid[-1]),
        jnp.interp(points, grid, values),
        value_basis * edge + slope_basis * increment,
    )


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_axis_and_inner_core_hold_first_sample(sign):
    assert jax.config.jax_enable_x64
    profile = SampledFluxFunction([0.05, 0.1, 0.5, 0.95], sign * np.array([7, 5, 3, 1]))
    points = jnp.array([-0.1, 0.0, 0.02, 0.049, 0.05], dtype=jnp.float64)
    np.testing.assert_array_equal(profile(points), np.full(points.shape, sign * 7))
    np.testing.assert_array_equal(jax.jit(lambda f, x: f(x))(profile, points), sign * 7)
    assert float(jax.grad(profile)(0.02)) == 0.0


@pytest.mark.parametrize("start", [0.0, 0.05])
def test_at_and_above_first_knot_is_bit_identical(start):
    profile = SampledFluxFunction(np.linspace(start, 1, 65), np.linspace(7, -2, 65))
    points = jnp.linspace(start, 1.04, 4001, dtype=jnp.float64)
    np.testing.assert_array_equal(profile(points), capped_reference(profile, points))
    np.testing.assert_array_equal(
        jax.jit(lambda f, x: f(x))(profile, points),
        jax.jit(capped_reference)(profile, points),
    )


def test_zero_start_holds_beyond_normalizing_axis():
    profile = SampledFluxFunction([0.0, 0.1, 1.0], [7, 5, 1])
    points = jnp.array([-1.0, -0.02, -1e-12, 0.0], dtype=jnp.float64)
    np.testing.assert_array_equal(profile(points), np.full(points.shape, 7.0))
    assert float(jax.grad(profile)(-0.02)) == 0.0


def test_upper_end_cap_is_bit_identical():
    profile = SampledFluxFunction([0.05, 0.1, 0.5, 0.95], [7, 5, 3, 1])
    points = jnp.linspace(0.95, 1.1, 4001, dtype=jnp.float64)
    expected = capped_reference(profile, points)
    np.testing.assert_array_equal(profile(points), expected)
    np.testing.assert_array_equal(
        jax.jit(lambda f, x: f(x))(profile, points),
        jax.jit(capped_reference)(profile, points),
    )
    assert float(profile(1.1)) == 0.0


def test_member_tables_remain_dynamic_array_leaves():
    profiles = [
        SampledFluxFunction([0.05, 0.1, 1.0], values)
        for values in ([7, 5, 1], [9, 4, 2])
    ]
    traces = []

    @jax.jit
    def evaluate(profile, points):
        traces.append(True)
        return profile(points)

    points = jnp.array([0.0, 0.02, 0.3, 1.02], dtype=jnp.float64)
    singles = [evaluate(profile, points) for profile in profiles]
    assert len(traces) == 1
    assert all(isinstance(leaf, jax.Array) for leaf in jax.tree.leaves(profiles[0]))
    stacked = jax.tree.map(lambda *leaves: jnp.stack(leaves), *profiles)
    batched = jax.jit(jax.vmap(lambda profile: profile(points)))(stacked)
    np.testing.assert_array_equal(batched, jnp.stack(singles))
    np.testing.assert_array_equal(batched[:, :2], [[7, 7], [9, 9]])


def test_saved_core_zoom_separates_every_maximum_and_saddle():
    from scripts.prototypes.diiid_core_hold.check_core_zoom import check_core_zoom

    figure = (
        Path(__file__).resolve().parents[1]
        / "docs/figures/sampled-profile-core-hold/frame-137-diverted-poloidal.svg"
    )
    result = check_core_zoom(figure)
    assert result["counts"] == {"maximum": 4, "saddle": 3}
    assert result["closest_pair"]["separation_in_marker_diameters"] >= 1.25
