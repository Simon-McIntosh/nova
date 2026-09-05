"""Structured topology reads consume the stationary rows their census authored."""

from __future__ import annotations

import numpy as np
import pytest

from nova.utilities.importmanager import skip_import

with skip_import("jax"):
    import jax
    import jax.numpy as jnp

    from nova.biot.null import Null1D, Null2D
    from nova.equilibrium import topology as topology_module
    from nova.equilibrium.conservation import FluxLattice
    from nova.equilibrium.flux_surface_connectivity import (
        fit_tensor_spline,
        polish_census_stationary_points,
    )
    from nova.equilibrium.forward_operator import _FixedDesignNull2D
    from nova.equilibrium.topology import Topology, TopologyClass
    from nova.geometry.hexstencil import hex_stencil
    from nova.jax.config import configure_dtypes


@pytest.fixture(scope="module", autouse=True)
def _double_precision():
    """Exercise the precision used by production forward maps."""
    configure_dtypes()


def _assert_bits(actual, expected, name: str) -> None:
    """Require equal shape, dtype and storage bytes for one telemetry field."""
    actual_array = np.asarray(actual)
    expected_array = np.asarray(expected)
    assert actual_array.shape == expected_array.shape, name
    assert actual_array.dtype == expected_array.dtype, name
    assert actual_array.tobytes() == expected_array.tobytes(), name


@pytest.mark.parametrize(
    ("radial_offset", "vertical_offset", "stationary_radius"),
    ((1.2, 0.04, 0.83), (1.16, -0.07, 0.71)),
)
def test_structured_read_consumes_retained_rows_without_repolishing(
    monkeypatch, radial_offset, vertical_offset, stationary_radius
):
    """Selected state and census telemetry survive the direct read bitwise."""
    radial = np.linspace(0.2, 2.2, 17)
    vertical = np.linspace(-1.2, 1.2, 19)
    lattice = FluxLattice(radial, vertical)
    locator = Null2D.from_coordinates(
        lattice.coordinate,
        hex_stencil(lattice.shape),
        maxsize=30,
    )
    fixed = _FixedDesignNull2D.from_locator(locator, extremum_polarity=-1)
    radial_grid, vertical_grid = np.meshgrid(radial, vertical)
    local_r = radial_grid - radial_offset
    values = (
        local_r**3 / 3.0
        - stationary_radius**2 * local_r
        + 0.7 * (vertical_grid - vertical_offset) ** 2
    )
    grid_flux = jnp.asarray(values.T.reshape(-1), dtype=jnp.float64)
    wall = jnp.asarray(((0.2, -1.2), (2.2, -1.2), (2.2, 1.2), (0.2, 1.2)))
    wall_local_r = wall[:, 0] - radial_offset
    wall_flux = (
        wall_local_r**3 / 3.0
        - stationary_radius**2 * wall_local_r
        + 0.7 * (wall[:, 1] - vertical_offset) ** 2
    )
    topology = Topology(fixed, Null1D(wall))
    polarity = jnp.asarray(-1)
    inside = jnp.ones(grid_flux.shape, dtype=bool)
    (vmap_o, vmap_x), census = fixed.read_census(grid_flux)
    data_w = topology.wall(wall_flux, polarity)
    surface = fit_tensor_spline(
        fixed.spline_radial,
        fixed.spline_vertical,
        jnp.asarray(values),
    )
    qualified_o = topology.qualified_o_candidates(
        vmap_o,
        vmap_x,
        data_w,
        polarity,
        grid_flux,
        inside,
        surface,
    )
    selection = topology.o_point_qualification(vmap_o, polarity, qualified_o)
    data_o = selection.data
    data_x = topology.x_point_data(vmap_x, polarity, data_o[2])
    selected_index = jnp.stack(
        (
            topology.o_point_index(vmap_o, polarity, qualified_o),
            topology.x_point_index(vmap_x, polarity, data_o[2]),
        )
    )
    expected_o, expected_x, previous_receipt = polish_census_stationary_points(
        jnp.asarray(values),
        fixed.spline_radial,
        fixed.spline_vertical,
        data_x[2],
        polarity,
        data_o,
        data_x,
        surface=surface,
    )

    def unexpected_polish(*_arguments, **_keywords):
        raise AssertionError("structured retained rows must not be polished twice")

    monkeypatch.setattr(
        topology_module, "polish_census_stationary_points", unexpected_polish
    )
    Topology.read_qualification.clear_cache()
    result = topology.read_qualification(
        jnp.r_[grid_flux, wall_flux],
        polarity,
        inside,
        int(TopologyClass.DIVERTED),
    )
    jax.block_until_ready(result)

    expected_stationary = jnp.stack((expected_o, expected_x))[:, :3]
    published_stationary = jnp.column_stack(
        (
            jnp.stack((result.state.axis, result.state.x_point)),
            jnp.stack((result.state.axis_flux, result.state.x_point_flux)),
        )
    )
    _assert_bits(published_stationary, expected_stationary, "published_stationary")
    for name in (
        "selected_position_rz",
        "selected_value",
        "local_value_evidence",
        "spline_authored",
    ):
        _assert_bits(result.polish_receipt[name], previous_receipt[name], name)

    kind = np.arange(2)
    index = np.asarray(selected_index)
    projected = {
        "gradient": census["retained_spline_gradient"][kind, index],
        "gradient_norm": census["retained_spline_gradient_norm"][kind, index],
        "hessian_determinant": census["retained_spline_hessian_determinant"][
            kind, index
        ],
        "representative_origin_index": census["retained_representative_origin_index"][
            kind, index
        ],
        "representative_origin_rz": census["retained_representative_origin_rz"][
            kind, index
        ],
        "multiplicity": census["retained_multiplicity"][kind, index],
        "requested_displacement": census["retained_requested_displacement"][
            kind, index
        ],
        "root_uncertainty": census["retained_root_uncertainty"][kind, index],
    }
    for name, expected in projected.items():
        _assert_bits(result.polish_receipt[name], expected, name)
