"""Analytic-separatrix authority for the closed-form fixture exterior."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from nova.equilibrium.forward_operator import ForwardFluxOperator
from nova.jax.config import configure_dtypes
from scripts.analytic_oracle_fixtures import measure as fixture
from tests.rotating_equilibrium_references import reference_cases


def test_clipped_analytic_moments_close_the_weak_fixture_without_a_topology_read(
    monkeypatch,
):
    """The explicit analytic clip closes its own map to binary64 roundoff."""
    configure_dtypes()
    assert jnp.asarray(1.0).dtype == jnp.float64
    case = reference_cases()["weak-rotation-reactor"].static_limit()
    machine = fixture.cached_machine(
        case,
        -110,
        wall_nodes=fixture.WALL_POINT_COUNT,
    )
    coordinates = np.vstack(
        (machine.node, machine.wall_node, machine.sample_coordinates)
    )
    exact = fixture.exact_state(case, coordinates)
    empty_operator = fixture.forward_operator(case, machine)

    def refuse_production_read(*_args, **_kwargs):
        raise AssertionError(
            "fixture construction consulted the production topology read"
        )

    with monkeypatch.context() as isolated:
        isolated.setattr(
            ForwardFluxOperator,
            "_fixed_design_read",
            refuse_production_read,
        )
        physical = fixture.exact_current_moments(
            case,
            empty_operator,
            exact,
            analytic=case,
        )

    coefficients = empty_operator.coupling_current_moments(physical)
    internal = fixture._internal_flux_image(empty_operator, coefficients)
    exterior = exact - internal
    operator = fixture.forward_operator(case, machine, exterior)
    mapped = np.asarray(operator.external()) + fixture._internal_flux_image(
        operator, coefficients
    )

    assert np.max(np.abs(mapped - exact)) <= 1.0e-12
    assert np.count_nonzero(np.asarray(physical.cell_current)) > 0
    whole = fixture.whole_cell_current_moments(case, empty_operator, exact)
    assert not np.array_equal(
        np.asarray(physical.cell_current), np.asarray(whole.cell_current)
    )
