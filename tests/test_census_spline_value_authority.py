"""Common tensor-spline authority for published census stationary points."""

from __future__ import annotations

import numpy as np
import pytest

from nova.utilities.importmanager import skip_import

with skip_import("jax"):
    import jax.numpy as jnp

    from benchmarks.solovev_certificate import AXIS_M, X_POINT_M, _case, _exact_state
    from nova.biot.null import Null1D, Null2D
    from nova.equilibrium.flux_surface_connectivity import (
        fit_tensor_spline,
        polish_census_stationary_points,
        polish_stationary_points,
    )
    from nova.equilibrium.topology import Topology, TopologyClass
    from nova.geometry.hexstencil import hex_stencil
    from nova.jax.config import configure_dtypes
    from scripts.analytic_oracle_fixtures import measure as oracle_fixture


@pytest.fixture(scope="module", autouse=True)
def _double_precision():
    """Match the precision used by the production topology read."""
    configure_dtypes()


def _selected_rows(spline, seeds):
    valid = jnp.ones(seeds.shape[:-1], dtype=bool)
    polished = polish_stationary_points(spline, seeds, valid, stationary_steps=32)
    return jnp.column_stack(
        (
            polished["position_rz"],
            polished["value"],
            jnp.zeros(seeds.shape[0], dtype=seeds.dtype),
        )
    )


def _cubic_stationary_field(radial, vertical):
    radial_grid, vertical_grid = jnp.meshgrid(radial, vertical)
    return radial_grid**3 / 3.0 - radial_grid + vertical_grid**2 / 2.0


def test_accepted_points_and_values_come_from_one_tensor_spline():
    """Local value evidence cannot replace either accepted spline value."""
    radial = jnp.linspace(-1.5, 1.5, 17)
    vertical = jnp.linspace(-1.2, 1.2, 17)
    values = _cubic_stationary_field(radial, vertical)
    surface = fit_tensor_spline(radial, vertical, values)
    selected = _selected_rows(
        surface,
        jnp.asarray(((0.9, 0.03), (-0.9, -0.03)), dtype=values.dtype),
    )

    extremum, saddle, receipt = polish_census_stationary_points(
        values,
        radial,
        vertical,
        selected[1, 2],
        jnp.asarray(-1.0, dtype=values.dtype),
        selected[0],
        selected[1],
    )
    published = jnp.stack((extremum, saddle))
    direct_value = surface(published[:, 0], published[:, 1])

    np.testing.assert_array_equal(np.asarray(receipt["spline_authored"]), True)
    np.testing.assert_allclose(
        np.asarray(published[:, :2]),
        np.asarray(receipt["selected_position_rz"]),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(published[:, 2]), np.asarray(direct_value), rtol=0.0, atol=2.0e-15
    )
    assert np.any(
        np.asarray(receipt["local_value_evidence"]) != np.asarray(published[:, 2])
    )


def test_topology_state_consumes_spline_authored_stationary_values():
    """The public topology state reads the values named by the polish receipt."""
    radial = np.linspace(-1.5, 1.5, 17)
    vertical = np.linspace(-1.2, 1.2, 17)
    radial_grid, vertical_grid = np.meshgrid(radial, vertical, indexing="ij")
    coordinates = np.c_[radial_grid.ravel(), vertical_grid.ravel()]
    grid = Null2D.from_coordinates(
        coordinates, hex_stencil(radial_grid.shape), maxsize=30
    )
    wall = np.asarray(((-1.5, -1.2), (1.5, -1.2), (1.5, 1.2), (-1.5, 1.2)))
    topology = Topology(grid, Null1D(jnp.asarray(wall)))
    grid_values = (radial_grid**3 / 3.0 - radial_grid + vertical_grid**2 / 2.0).ravel()
    wall_values = wall[:, 0] ** 3 / 3.0 - wall[:, 0] + wall[:, 1] ** 2 / 2.0
    result = topology.read_qualification(
        jnp.asarray(np.r_[grid_values, wall_values]),
        -1,
        jnp.ones(len(coordinates), dtype=bool),
        int(TopologyClass.DIVERTED),
    )
    surface = fit_tensor_spline(
        jnp.asarray(radial),
        jnp.asarray(vertical),
        jnp.asarray(grid_values.reshape(radial_grid.shape).T),
    )
    positions = jnp.stack((result.state.axis, result.state.x_point))
    state_values = jnp.stack((result.state.axis_flux, result.state.x_point_flux))

    np.testing.assert_array_equal(
        np.asarray(result.polish_receipt["spline_authored"]), True
    )
    np.testing.assert_allclose(
        np.asarray(state_values),
        np.asarray(surface(positions[:, 0], positions[:, 1])),
        rtol=0.0,
        atol=2.0e-15,
    )
    np.testing.assert_allclose(
        np.asarray(state_values),
        np.asarray(result.polish_receipt["selected_value"]),
        rtol=0.0,
        atol=0.0,
    )


def test_diverted_exact_oracle_value_error_favours_tensor_spline():
    """The complete 136-sample oracle map reports both value errors."""
    carrier_case, _source_case, exact = _case("diverted-jump-bearing")
    machine = oracle_fixture.cached_machine(
        carrier_case,
        -110,
        wall_nodes=oracle_fixture.WALL_POINT_COUNT,
    )
    assert len(machine.node) == 136
    radial = jnp.linspace(float(machine.node[:, 0].min()), machine.node[:, 0].max(), 17)
    vertical = jnp.linspace(
        float(machine.node[:, 1].min()), machine.node[:, 1].max(), 8
    )
    radial_grid, vertical_grid = jnp.meshgrid(radial, vertical)
    coordinates = np.c_[
        np.asarray(radial_grid).ravel(), np.asarray(vertical_grid).ravel()
    ]
    values = jnp.asarray(
        _exact_state("diverted-jump-bearing", exact, coordinates).reshape((8, 17))
    )
    surface = fit_tensor_spline(radial, vertical, values)
    selected = _selected_rows(
        surface,
        jnp.asarray(np.stack((AXIS_M + (0.01, 0.01), X_POINT_M + (0.01, 0.01)))),
    )

    extremum, saddle, receipt = polish_census_stationary_points(
        values,
        radial,
        vertical,
        selected[1, 2],
        jnp.asarray(1.0, dtype=values.dtype),
        selected[0],
        selected[1],
    )
    published = jnp.stack((extremum, saddle))
    closed_form = _exact_state(
        "diverted-jump-bearing", exact, np.asarray(published[:, :2])
    )
    tensor_error = np.abs(np.asarray(published[:, 2]) - closed_form)
    local_error = np.abs(np.asarray(receipt["local_value_evidence"]) - closed_form)
    print(
        "diverted_exact_value_error "
        f"tensor_spline={tensor_error.tolist()} "
        f"seven_point_local={local_error.tolist()}"
    )

    np.testing.assert_array_equal(np.asarray(receipt["spline_authored"]), True)
    np.testing.assert_allclose(
        np.asarray(published[:, 2]),
        np.asarray(surface(published[:, 0], published[:, 1])),
        rtol=0.0,
        atol=2.0e-15,
    )
    assert np.all(tensor_error < local_error)
