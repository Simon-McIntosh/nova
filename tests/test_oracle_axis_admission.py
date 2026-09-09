"""Production topology admission on the coarse diverted analytic oracle."""

from __future__ import annotations

import numpy as np
import pytest

from nova.utilities.importmanager import skip_import

with skip_import("jax"):
    import jax.numpy as jnp

    from benchmarks.solovev_certificate import (
        AXIS_M,
        DIVERTED_CASE_NAME,
        X_POINT_M,
        _case,
        _case_machine,
        _diverted_geometry_row,
        _exact_state,
    )
    from nova.jax.config import configure_dtypes
    from scripts.analytic_oracle_fixtures import measure as oracle_fixture


@pytest.fixture(scope="module", autouse=True)
def _double_precision():
    """Match the precision used by the production certificate."""
    configure_dtypes()


def _diverted_exact_read(requested_cells):
    """Build a certificate carrier and read its closed-form flux field."""
    carrier_case, source_case, exact = _case(DIVERTED_CASE_NAME)
    machine = _case_machine(DIVERTED_CASE_NAME, carrier_case, exact, requested_cells)
    coordinates = np.vstack(
        (machine.node, machine.wall_node, machine.sample_coordinates)
    )
    oracle_state = _exact_state(DIVERTED_CASE_NAME, exact, coordinates)
    empty_operator = oracle_fixture.forward_operator(source_case, machine)
    exact_physical = oracle_fixture.exact_current_moments(
        source_case, empty_operator, oracle_state
    )
    exact_coefficients = empty_operator.coupling_current_moments(exact_physical)
    exact_internal = oracle_fixture._internal_flux_image(
        empty_operator, exact_coefficients
    )
    operator = oracle_fixture.forward_operator(
        source_case, machine, oracle_state - exact_internal
    )
    _masks, state = operator.read(jnp.asarray(oracle_state))
    grid_flux = jnp.asarray(oracle_state[: len(machine.node)])
    table_status = operator._fixed_design_topology.grid.candidate_table_status(
        grid_flux
    )
    return machine, operator, state, table_status


def test_coarse_diverted_exact_field_admits_axis_and_saddle():
    """The exact diverted field resolves both nulls on its certificate ladder."""
    coarse_machine, coarse_operator, coarse_state, coarse_status = _diverted_exact_read(
        -110
    )
    coarse_pitch = float(np.sqrt(np.median(np.asarray(coarse_machine.area))))
    coarse_grid = coarse_operator._fixed_design_topology.grid
    nearest_saddle_cell = int(
        np.argmin(
            np.linalg.norm(
                np.asarray(coarse_grid.locator.physical_origin) - X_POINT_M,
                axis=1,
            )
        )
    )

    assert len(coarse_machine.node) > 0
    assert len(coarse_machine.wall_node) == oracle_fixture.WALL_POINT_COUNT
    carrier_case, _source_case, _exact = _case(DIVERTED_CASE_NAME)
    default_identity = oracle_fixture.cache_identity(
        carrier_case,
        requested_cells=-110,
        wall_nodes=oracle_fixture.WALL_POINT_COUNT,
    )
    explicit_identity = oracle_fixture.cache_identity(
        carrier_case,
        requested_cells=-110,
        wall_nodes=oracle_fixture.WALL_POINT_COUNT,
        wall=coarse_machine.wall_node,
    )
    assert explicit_identity["discretisation"]["wall_source"] == "explicit"
    assert (
        explicit_identity["discretisation"]["wall_content"]["sha256"]
        != default_identity["discretisation"]["wall_content"]["sha256"]
    )
    np.testing.assert_allclose(np.asarray(coarse_state.axis), AXIS_M, atol=coarse_pitch)
    assert np.all(np.isnan(np.asarray(coarse_state.x_point)))
    assert not bool(coarse_state.diverted)
    assert int(coarse_status["ring_crossing_count"][nearest_saddle_cell]) == 2
    assert bool(coarse_status["ring_resolution_limited"][nearest_saddle_cell])
    assert not np.any(np.asarray(coarse_status["truncated"]))

    # The 136-cell rung resolves only the axis; containment first resolves X at 342.
    fine_machine, _fine_operator, fine_state, fine_status = _diverted_exact_read(-342)
    fine_pitch = float(np.sqrt(np.median(np.asarray(fine_machine.area))))

    assert len(fine_machine.node) > len(coarse_machine.node)
    np.testing.assert_allclose(np.asarray(fine_state.axis), AXIS_M, atol=fine_pitch)
    np.testing.assert_allclose(
        np.asarray(fine_state.x_point), X_POINT_M, atol=fine_pitch
    )
    assert bool(fine_state.diverted)
    assert not np.any(np.asarray(fine_status["truncated"]))


@pytest.mark.parametrize("requested_cells", (-110, -342))
def test_diverted_exact_boundary_clears_axis_and_matches_contour(requested_cells):
    """The analytic separatrix encloses a resolved core inside the machine wall."""
    geometry = _diverted_geometry_row(requested_cells)

    assert geometry["inboard_separatrix_radius_m"] >= 0.85
    assert geometry["axis_clearance_fraction_of_minor_radius"] >= 0.8
    assert geometry["x_point_below_magnetic_axis"]
    assert geometry["x_point_inside_wall_polygon"]
    assert geometry["separatrix_wall_clearance_fraction_of_minor_radius"] >= 0.3
    assert geometry["analytic_grad_shafranov_residual_relative"] < 1.0e-10
    assert geometry["hausdorff_distance_m"] < geometry["characteristic_cell_pitch_m"]
