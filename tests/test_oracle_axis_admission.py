"""Production topology admission on the coarse diverted analytic oracle."""

from __future__ import annotations

import numpy as np
import pytest

from nova.utilities.importmanager import skip_import

with skip_import("jax"):
    import jax.numpy as jnp

    from benchmarks.solovev_certificate import (
        AXIS_M,
        X_POINT_M,
        _case,
        _exact_state,
    )
    from nova.jax.config import configure_dtypes
    from scripts.analytic_oracle_fixtures import measure as oracle_fixture


@pytest.fixture(scope="module", autouse=True)
def _double_precision():
    """Match the precision used by the production certificate."""
    configure_dtypes()


def _coarse_diverted_exact_read():
    """Build the 136-cell carrier and read its closed-form flux field."""
    carrier_case, source_case, exact = _case("diverted-jump-bearing")
    machine = oracle_fixture.cached_machine(
        carrier_case,
        -110,
        wall_nodes=oracle_fixture.WALL_POINT_COUNT,
    )
    coordinates = np.vstack(
        (machine.node, machine.wall_node, machine.sample_coordinates)
    )
    oracle_state = _exact_state("diverted-jump-bearing", exact, coordinates)
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
    return machine, state, table_status


def test_coarse_diverted_exact_field_admits_axis_and_saddle():
    """The 136-cell analytic field must read both declared stationary points."""
    machine, state, table_status = _coarse_diverted_exact_read()
    pitch = float(np.sqrt(np.median(np.asarray(machine.area))))

    assert len(machine.node) == 136
    np.testing.assert_allclose(np.asarray(state.axis), AXIS_M, atol=pitch)
    np.testing.assert_allclose(np.asarray(state.x_point), X_POINT_M, atol=pitch)
    assert bool(state.diverted)
    assert not np.any(np.asarray(table_status["truncated"]))
