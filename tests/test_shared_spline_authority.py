"""One fitted tensor spline supplies every complete-map topology consumer."""

from __future__ import annotations

import numpy as np
import pytest

from nova.utilities.importmanager import skip_import

with skip_import("jax"):
    import jax
    import jax.numpy as jnp

    from nova.biot.null import Null1D, Null2D
    from nova.equilibrium import connectivity_boundary as boundary_module
    from nova.equilibrium import flux_surface_connectivity as surface_module
    from nova.equilibrium import topology as topology_module
    from nova.equilibrium.topology import Topology, TopologyClass
    from nova.geometry.hexstencil import hex_stencil
    from nova.jax.config import configure_dtypes


@pytest.fixture(scope="module", autouse=True)
def _double_precision():
    configure_dtypes()


def _structured_case():
    radial = jnp.linspace(-1.5, 1.5, 17)
    vertical = jnp.linspace(-1.2, 1.2, 17)
    radial_grid, vertical_grid = jnp.meshgrid(radial, vertical, indexing="ij")
    coordinate = np.c_[
        np.asarray(radial_grid).ravel(), np.asarray(vertical_grid).ravel()
    ]
    values = radial_grid**3 / 3.0 - radial_grid + vertical_grid**2 / 2.0
    wall = jnp.asarray(((-1.5, -1.2), (1.5, -1.2), (1.5, 1.2), (-1.5, 1.2)))
    wall_values = wall[:, 0] ** 3 / 3.0 - wall[:, 0] + wall[:, 1] ** 2 / 2.0
    topology = Topology(
        Null2D.from_coordinates(
            coordinate,
            hex_stencil(radial_grid.shape),
            maxsize=30,
        ),
        Null1D(wall),
    )
    complete = jnp.r_[values.ravel(), wall_values]
    return radial, vertical, values.T, wall, topology, complete


def _clear_topology_caches():
    Topology.read_qualification.clear_cache()
    Topology.qualified_o_candidates.clear_cache()
    Topology.axis_component.clear_cache()
    surface_module.polish_census_stationary_points.clear_cache()
    surface_module.hex_edge_admissibility.clear_cache()


def test_structured_topology_read_fits_one_tensor_spline(monkeypatch):
    radial, vertical, _values, _wall, topology, complete = _structured_case()
    original_fit = topology_module.fit_tensor_spline
    fit_count = 0

    def counted_fit(*arguments):
        nonlocal fit_count
        fit_count += 1
        return original_fit(*arguments)

    def unexpected_fit(*_arguments, **_keywords):
        raise AssertionError("a topology consumer refitted the complete map")

    monkeypatch.setattr(topology_module, "fit_tensor_spline", counted_fit)
    monkeypatch.setattr(surface_module, "fit_tensor_spline", unexpected_fit)
    _clear_topology_caches()
    result = topology.read_qualification(
        complete,
        -1,
        jnp.ones(17 * 17, dtype=bool),
        int(TopologyClass.DIVERTED),
    )
    jax.block_until_ready(result.state.axis_flux)

    assert fit_count == 1
    direct_surface = original_fit(radial, vertical, _values)
    positions = jnp.stack((result.state.axis, result.state.x_point))
    direct = direct_surface(positions[:, 0], positions[:, 1])
    published = jnp.stack((result.state.axis_flux, result.state.x_point_flux))
    np.testing.assert_allclose(np.asarray(published), np.asarray(direct), atol=2.0e-15)
    np.testing.assert_allclose(
        np.asarray(published),
        np.asarray(result.polish_receipt["selected_value"]),
        rtol=0.0,
        atol=0.0,
    )
    assert direct_surface.radial.shape == radial.shape
    assert direct_surface.vertical.shape == vertical.shape


def test_shared_surface_supplies_polish_wall_contour_and_edge_clips(monkeypatch):
    radial, vertical, values, wall, _topology, _complete = _structured_case()
    surface = surface_module.fit_tensor_spline(radial, vertical, values)
    seeds = jnp.asarray(((1.0, 0.0), (-1.0, 0.0)), dtype=values.dtype)
    polished = surface_module.polish_stationary_points(
        surface,
        seeds,
        jnp.ones(2, dtype=bool),
        stationary_steps=16,
    )
    selected = jnp.column_stack(
        (
            polished["position_rz"],
            polished["value"],
            jnp.zeros(2, dtype=values.dtype),
        )
    )

    def unexpected_fit(*_arguments, **_keywords):
        raise AssertionError("a shared-surface consumer refitted the complete map")

    monkeypatch.setattr(surface_module, "fit_tensor_spline", unexpected_fit)
    monkeypatch.setattr(boundary_module, "fit_tensor_spline", unexpected_fit)
    surface_module.polish_census_stationary_points.clear_cache()
    surface_module.traced_spline_contour.clear_cache()
    surface_module.hex_edge_admissibility.clear_cache()
    boundary_module.traced_boundary_read.clear_cache()

    extremum, saddle, receipt = surface_module.polish_census_stationary_points(
        values,
        radial,
        vertical,
        selected[1, 2],
        jnp.asarray(-1.0, dtype=values.dtype),
        selected[0],
        selected[1],
        surface=surface,
    )
    published = jnp.stack((extremum, saddle))
    np.testing.assert_allclose(
        np.asarray(published[:, 2]),
        np.asarray(surface(published[:, 0], published[:, 1])),
        atol=2.0e-15,
    )
    np.testing.assert_allclose(
        np.asarray(receipt["selected_value"]),
        np.asarray(published[:, 2]),
        rtol=0.0,
        atol=0.0,
    )

    inside = jnp.ones_like(values, dtype=bool)
    wall_flux = surface(wall[:, 0], wall[:, 1])
    boundary = boundary_module.traced_boundary_read(
        values,
        radial,
        vertical,
        inside,
        extremum[0],
        extremum[1],
        12,
        8,
        2,
        jnp.empty((0,), dtype=values.dtype),
        jnp.asarray(0.999, dtype=values.dtype),
        wall[:, 0],
        wall[:, 1],
        wall_flux,
        surface=surface,
    )
    np.testing.assert_allclose(
        np.asarray(boundary["psi_axis"]),
        np.asarray(surface(extremum[0], extremum[1])),
        atol=2.0e-15,
    )
    np.testing.assert_allclose(
        np.asarray(boundary["limiter_psi"]),
        np.asarray(surface(boundary["limiter_r"], boundary["limiter_z"])),
        atol=2.0e-15,
    )

    contour = surface_module.traced_spline_contour(
        values,
        radial,
        vertical,
        jnp.asarray(0.0, dtype=values.dtype),
        bisection_steps=24,
        surface=surface,
    )
    crossing = np.asarray(contour["edge_crossing"])
    crossing_points = contour["edge_crossing_rz"]
    crossing_values = np.asarray(
        surface(crossing_points[..., 0], crossing_points[..., 1])
    )
    assert np.any(crossing)
    np.testing.assert_allclose(crossing_values[crossing], 0.0, atol=2.0e-8)

    edge_start = jnp.asarray((-1.4, -0.6), dtype=values.dtype)
    edge_end = jnp.asarray((1.4, 0.6), dtype=values.dtype)
    shared_edges = jnp.broadcast_to(
        jnp.stack((edge_start, edge_end)),
        (1, 7, 2, 2),
    )
    links = surface_module.hex_edge_admissibility(
        values,
        radial,
        vertical,
        jnp.asarray(0.0, dtype=values.dtype),
        surface(1.0, 0.0),
        shared_edges,
        surface=surface,
    )
    assert links.shape == (1, 7)
    assert bool(links[0, 0])
