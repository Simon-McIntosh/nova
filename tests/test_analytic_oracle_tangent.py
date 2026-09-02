"""Derivative sentry for the static analytic-oracle source."""

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks.nan_tangent_localisation import (
    CASE_NAME,
    REQUESTED_CELLS,
    _probe_direction,
)
from benchmarks.solovev_certificate import (
    _case,
    _closed_form_current_target,
    _exact_state,
)
from nova.equilibrium import ForwardProfile
from nova.equilibrium.stencil_mesh import StencilMesh
from nova.jax.config import configure_dtypes
from scripts.analytic_oracle_fixtures import measure as oracle_fixture
from scripts.oracle_rebaseline import measure as recovery


def test_static_oracle_current_moment_jvp_is_finite():
    """The reduced static source has a finite current-moment linearisation."""
    configure_dtypes()

    carrier_case, source_case, exact = _case(CASE_NAME)
    machine = oracle_fixture.cached_machine(
        carrier_case,
        REQUESTED_CELLS,
        wall_nodes=oracle_fixture.WALL_POINT_COUNT,
    )
    coordinates = np.vstack(
        (machine.node, machine.wall_node, machine.sample_coordinates)
    )
    state = jnp.asarray(_exact_state(CASE_NAME, exact, coordinates))
    empty_operator = oracle_fixture.forward_operator(source_case, machine)
    exact_physical = oracle_fixture.exact_current_moments(
        source_case, empty_operator, np.asarray(state)
    )
    coefficients = empty_operator.coupling_current_moments(exact_physical)
    internal = oracle_fixture._internal_flux_image(empty_operator, coefficients)
    operator = oracle_fixture.forward_operator(
        source_case, machine, np.asarray(state) - internal
    )
    profile = ForwardProfile(
        operator,
        StencilMesh(machine.node, machine.stencil, machine.area),
        newton_steps=recovery.NEWTON_STEPS,
    )
    target_current, _centroid, _receipt = _closed_form_current_target(
        CASE_NAME, source_case, operator, exact_physical
    )
    direction, _direction_receipt = _probe_direction(profile, state, target_current)
    moments, tangent = jax.jvp(
        lambda candidate: jnp.stack(operator.cell_current_moments(candidate)),
        (state,),
        (direction,),
    )

    assert CASE_NAME == "strong-rotation-compact-static"
    assert REQUESTED_CELLS == -110
    assert jnp.all(jnp.isfinite(moments))
    assert jnp.all(jnp.isfinite(tangent))
