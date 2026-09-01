"""Checks for census-owned null selection with split-spline polishing."""

import jax.numpy as jnp
import numpy as np
import pytest

from benchmarks.hex_cell_field_feasibility import (
    AXIS,
    LOBE_OFFSET,
    SADDLE,
    _base_flux,
    hex_lattice,
    solovev_flux,
)
from nova.equilibrium.flux_surface_connectivity import (
    polish_census_stationary_points,
)
from nova.equilibrium.topology import TopologyClass
from nova.jax.config import configure_dtypes
from nova.linalg.split_spline import fit_split_spline
from tests.test_hex_flood_sn_secondary import (
    GEOMETRIES,
    GRID_SIZES,
    _topology_and_flux,
)


configure_dtypes()


def _carrier(shape: tuple[int, int]):
    centres, _, _ = hex_lattice(shape)
    radial = jnp.asarray(centres[..., 0])
    vertical = jnp.asarray(centres[..., 1])
    points = jnp.asarray(centres.reshape(-1, 2))
    values = solovev_flux(points).reshape(shape)
    level_set = (_base_flux(points) - LOBE_OFFSET**4).reshape(shape)
    return radial, vertical, values, level_set


def _selected_row(position: np.ndarray) -> jnp.ndarray:
    point = jnp.asarray(position)
    return jnp.r_[point, solovev_flux(point), 0.0]


def test_one_step_refines_only_the_two_census_selected_nulls():
    radial, vertical, values, _level_set = _carrier((41, 49))
    extremum_seed = _selected_row(AXIS + (0.005, 0.0))
    saddle_seed = _selected_row(SADDLE)
    boundary_value = solovev_flux(jnp.asarray(SADDLE))

    extremum, saddle, receipt = polish_census_stationary_points(
        values,
        radial,
        vertical,
        boundary_value,
        jnp.asarray(-1.0),
        extremum_seed,
        saddle_seed,
    )
    error = jnp.linalg.norm(
        jnp.stack((extremum[:2], saddle[:2])) - jnp.asarray((AXIS, SADDLE)),
        axis=-1,
    )

    print(
        f"solovev axis_error_m={float(error[0]):.12e} "
        f"saddle_error_m={float(error[1]):.12e} "
        f"iterations={np.asarray(receipt['iteration_count']).tolist()} "
        f"accepted={np.asarray(receipt['converged']).tolist()} "
        f"normalized_gradient={np.asarray(receipt['normalized_gradient']).tolist()} "
        f"tolerance={float(receipt['stationarity_tolerance'][0]):.12e}"
    )
    assert error[0] <= 2.0e-4
    assert error[1] <= 1.0e-12
    assert receipt["converged"].tolist() == [True, True]
    assert receipt["iteration_count"].tolist() == [1, 0]
    assert receipt["seed_stationary"].tolist() == [False, True]
    assert receipt["value"].shape == (2,)
    assert receipt["gradient"].shape == (2, 2)
    assert receipt["hessian"].shape == (2, 2, 2)
    np.testing.assert_array_equal(
        np.asarray(receipt["census_position_rz"]),
        np.asarray(jnp.stack((extremum_seed[:2], saddle_seed[:2]))),
    )


def test_sample_coincidence_does_not_imply_stationarity():
    topology, psi, inside, _radius, _height = _topology_and_flux(GEOMETRIES[0], 33)
    grid_flux, wall_flux = topology.split_flux_map(psi)
    extrema, saddles = topology.grid(grid_flux)
    wall = topology.wall(wall_flux, -1)
    qualified = topology.qualified_o_candidates(
        extrema, saddles, wall, -1, grid_flux, inside
    )
    extremum = topology.o_point_qualification(extrema, -1, qualified).data
    saddle = topology.x_point_data(saddles, -1, extremum[2])
    coordinate = topology.grid.coordinate
    displaced_target = saddle[:2] + jnp.asarray((0.0, 0.15))
    sample_index = jnp.argmin(jnp.linalg.norm(coordinate - displaced_target, axis=-1))
    sampled_saddle = saddle.at[:2].set(coordinate[sample_index])
    sampled_saddle = sampled_saddle.at[2].set(grid_flux[sample_index])
    radial_count = topology.connectivity_radius.size
    vertical_count = topology.connectivity_height.size
    values = grid_flux.reshape((radial_count, vertical_count)).T

    _selected_extremum, selected_saddle, receipt = polish_census_stationary_points(
        values,
        topology.connectivity_radius,
        topology.connectivity_height,
        sampled_saddle[2],
        jnp.asarray(-1.0),
        extremum,
        sampled_saddle,
    )

    assert float(jnp.linalg.norm(sampled_saddle[:2] - coordinate[sample_index])) == 0.0
    assert not bool(receipt["seed_stationary"][1])
    assert not bool(receipt["converged"][1])
    assert float(receipt["seed_normalized_gradient"][1]) > float(
        receipt["stationarity_tolerance"][1]
    )
    np.testing.assert_array_equal(
        np.asarray(selected_saddle), np.asarray(sampled_saddle)
    )


def test_failed_fit_retains_the_census_rows_and_reports_failure():
    radial, vertical, values, _level_set = _carrier((13, 15))
    extremum_seed = _selected_row(AXIS + (0.005, 0.0))
    saddle_seed = _selected_row(SADDLE)

    extremum, saddle, receipt = polish_census_stationary_points(
        values.at[0, 0].set(jnp.nan),
        radial,
        vertical,
        solovev_flux(jnp.asarray(SADDLE)),
        jnp.asarray(-1.0),
        extremum_seed,
        saddle_seed,
    )

    np.testing.assert_array_equal(np.asarray(extremum), np.asarray(extremum_seed))
    np.testing.assert_array_equal(np.asarray(saddle), np.asarray(saddle_seed))
    assert receipt["fit_converged"].tolist() == [False, False]
    assert receipt["converged"].tolist() == [False, False]


def test_limited_read_fits_at_the_wall_contact_level():
    topology, psi, inside, _radius, _height = _topology_and_flux(GEOMETRIES[0], 33)
    grid_flux, _wall_flux = topology.split_flux_map(psi)
    extrema, saddles = topology.grid(grid_flux)
    provisional_extremum = topology.o_point_data(extrema, -1)
    census_saddle = topology.x_point_data(saddles, -1, provisional_extremum[2])
    result = topology.read_qualification(psi, -1, inside, int(TopologyClass.LIMITED))

    np.testing.assert_allclose(
        np.asarray(result.polish_receipt["interface_value"]),
        np.asarray((result.state.wall_point_flux, result.state.wall_point_flux)),
        rtol=0.0,
        atol=0.0,
    )
    assert not np.isclose(
        float(result.state.wall_point_flux),
        float(census_saddle[2]),
        rtol=0.0,
        atol=1e-8,
    )


@pytest.mark.parametrize("size", GRID_SIZES)
@pytest.mark.parametrize("geometry", GEOMETRIES, ids=lambda item: item.name)
def test_secondary_null_polish_receipt_is_explicit(geometry, size):
    topology, psi, inside, _radius, _height = _topology_and_flux(geometry, size)
    result = topology.read_qualification(psi, -1, inside, int(TopologyClass.DIVERTED))
    receipt = result.polish_receipt
    accepted = np.asarray(receipt["converged"]).tolist()
    normalized_gradient = np.asarray(receipt["normalized_gradient"]).tolist()
    normalized_value_change = np.asarray(
        receipt["normalized_value_change"]
    ).tolist()
    tolerance = np.asarray(receipt["stationarity_tolerance"]).tolist()
    fit_converged = np.asarray(receipt["fit_converged"]).tolist()
    print(
        f"secondary_null_receipt case={geometry.name}-{size} "
        f"fit_converged={fit_converged} accepted={accepted} "
        f"normalized_gradient={normalized_gradient} "
        f"normalized_value_change={normalized_value_change} tolerance={tolerance}"
    )

    assert fit_converged == [True, True]
    assert np.all(np.isfinite(normalized_gradient))
    assert np.all(np.isfinite(normalized_value_change))
    assert np.all(np.asarray(tolerance) > 0.0)
    retained = ~np.asarray(receipt["converged"])
    np.testing.assert_array_equal(
        np.asarray(receipt["selected_position_rz"])[retained],
        np.asarray(receipt["census_position_rz"])[retained],
    )


def test_production_capacity_fit_receipts_report_solve_convergence():
    receipts = []
    for shape in ((33, 33), (65, 65)):
        radial, vertical, values, level_set = _carrier(shape)
        spline = fit_split_spline(radial, vertical, values, level_set)
        receipts.append(
            (
                shape,
                int(spline.solve_iterations),
                float(spline.solve_residual),
                bool(spline.solve_converged),
            )
        )

    print(f"production_fit_receipts={receipts}")
    assert [row[1] for row in receipts] == [1, 1]
    assert all(row[2] < 1.0e-10 for row in receipts)
    assert all(row[3] for row in receipts)
